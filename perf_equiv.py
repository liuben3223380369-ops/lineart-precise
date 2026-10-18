"""本轮性能优化的数值等价性验证：新旧实现对比。"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/workspace/lineart-precise")
from lineart.adaptive import sauvola_threshold
from lineart.background import detect_background
from lineart.edges import _otsu_hi, verify_with_gradient
from lineart.guided import edge_preserving_denoise, guided_filter
from lineart.illumination import estimate_background, normalize_illumination
from lineart.scenes import _is_pixel_art

rng = np.random.default_rng(42)
img = cv2.imread("/workspace/lineart-precise/demo/samples/sketch_photo.png", cv2.IMREAD_COLOR)
big = cv2.resize(img, (2048, 1536), interpolation=cv2.INTER_CUBIC)
gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
print("test img:", gray_big.shape)


# ---------- 旧实现复刻 ----------
def guided_filter_old(guide, src, radius=3, eps=1e-4):
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(a):
        return cv2.boxFilter(a, -1, ksize, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(guide * src)
    corr_ii = box(guide * guide)
    var_i = np.maximum(corr_ii - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def sauvola_threshold_old(gray, window=None, k=0.2, r_dynamic=0.5):
    g = gray.astype(np.float64)
    h, w = g.shape
    if window is None:
        window = int(np.clip(round(min(h, w) / 20.0), 15, 75))
    window = max(3, int(window) | 1)
    r = window // 2
    gp = cv2.copyMakeBorder(g, r, r, r, r, cv2.BORDER_REPLICATE)
    ii = cv2.integral(gp)
    ii2 = cv2.integral(gp * gp)
    area = float(window * window)
    so = (slice(window, window + h), slice(window, window + w))
    st = (slice(0, h), slice(window, window + w))
    sl = (slice(window, window + h), slice(0, w))
    sc = (slice(0, h), slice(0, w))
    s1 = ii[so] - ii[st] - ii[sl] + ii[sc]
    s2 = ii2[so] - ii2[st] - ii2[sl] + ii2[sc]
    m = s1 / area
    var = s2 / area - m * m
    s = np.sqrt(np.maximum(var, 0.0))
    t = m * (1.0 + k * (s / r_dynamic - 1.0))
    return t.astype(np.float32), m.astype(np.float32), s.astype(np.float32)


def estimate_background_old(gray, downscale=0.25, sigma_ratio=0.035):
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, int(round(w * downscale))),
                              max(1, int(round(h * downscale)))),
                       interpolation=cv2.INTER_AREA)
    sigma = max(1.0, max(small.shape) * sigma_ratio)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigma)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigma)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 1e-3, None)


def normalize_illumination_old(gray, strength=1.0):
    if strength <= 0:
        return gray.copy()
    bg = estimate_background_old(gray)
    corrected = gray / bg
    lo, hi = corrected.min(), corrected.max()
    if hi - lo < 1e-6:
        return gray.copy()
    corrected = (corrected - lo) / (hi - lo)
    blend = strength * corrected + (1.0 - strength) * gray
    return np.clip(blend, 0.0, 1.0).astype(np.float32)


def verify_with_gradient_old(ink, mag, gamma=0.7, gain=6.0, spread=5, noise_k=1.8):
    if spread >= 3:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spread, spread))
        ink = cv2.dilate(ink.astype(np.float32), kernel)
    from lineart.edges import _fast_median, _fast_quantile, _gamma_lut, _GAMMA_LUT_SIZE, EPS
    sub = mag[::4, ::4]
    noise = _fast_median(sub)
    floor = noise * noise_k + EPS
    ref = max(_fast_quantile(sub, 0.995), floor * 2.0)
    g = np.clip((mag - floor) / max(ref - floor, EPS), 0.0, 1.0)
    lut = _gamma_lut(gamma)
    idx = (g * (_GAMMA_LUT_SIZE - 1)).astype(np.int32)
    gate = np.clip(gain * lut[idx], 0.0, 1.0)
    return np.clip(ink * gate, 0.0, 1.0)


def _otsu_hi_old(strength):
    u8 = np.clip(strength * 255.0, 0, 255).astype(np.uint8)
    hi_val = cv2.threshold(u8, 0, 255, cv2.THRESH_OTSU)[0]
    if hi_val < 30:
        hi_val = 30
    return hi_val / 255.0


def _is_pixel_art_old(img_bgr, max_side=384):
    from lineart.scenes import _shrink_nn
    small = _shrink_nn(img_bgr, max_side)
    h, w = small.shape[:2]
    if h < 8 or w < 8:
        return 0.0
    q = (small.astype(np.int32) >> 4)
    n_colors = len(np.unique(q.reshape(-1, q.shape[2]), axis=0))
    if n_colors > 32:
        return 0.0
    f = small.astype(np.int16)
    dh = np.abs(np.diff(f, axis=1)).max(axis=2).ravel()
    dv = np.abs(np.diff(f, axis=0)).max(axis=2).ravel()
    d = np.concatenate([dh, dv])
    nz = d[d > 4]
    if len(nz) < 16:
        return 0.0
    mid_ratio = float(((nz >= 8) & (nz <= 90)).mean())
    hard_ratio = float((nz > 160).mean())
    if mid_ratio >= 0.35 or hard_ratio < 0.25:
        return 0.0
    conf = min(1.0, (0.35 - mid_ratio) * 2.0 + hard_ratio)
    return min(1.0, conf)


def detect_background_old(img_bgr):
    if img_bgr.shape[2] == 4:
        alpha = img_bgr[:, :, 3]
        if alpha.mean() < 200:
            return "transparent", False
    hsv = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = (h_ch >= 95) & (h_ch <= 135) & (s_ch > 70) & (v_ch > 30)
    bright_line = (v_ch > 170) & (s_ch < 90)
    if blue.mean() > 0.30 and bright_line.mean() > 0.015:
        return "blueprint", False
    from lineart.background import _corner_stats
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    corner = _corner_stats(gray)
    corner_mean = corner.mean()
    corner_std = corner.std()
    overall_mean = gray.mean()
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [8], [0, 255])
    hist = hist.ravel() / hist.sum()
    dark_bin = hist[:2].sum()
    bright_bin = hist[-2:].sum()
    if corner_std < 12 and corner_mean > 200 and bright_bin > 0.6:
        return "white", False
    if corner_std < 12 and corner_mean < 40 and dark_bin > 0.4:
        return "black", True
    if corner_std < 20 and 60 < corner_mean < 190:
        return "gray", False
    if corner_std > 30 and overall_mean > 120 and dark_bin > 0.15:
        return "white", False
    if corner_std > 30 and overall_mean < 100 and bright_bin > 0.15:
        return "black", True
    if dark_bin > 0.45 and bright_bin > 0.45:
        return "mixed", False
    if corner_mean < 80:
        return "black", True
    if corner_mean > 170:
        return "white", False
    return "gray", False


def report(name, diff, tol):
    ok = diff <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max|Δ|={diff:.3e} (tol {tol:.0e})")
    return ok


all_ok = True

# 1. guided 自引导
for trial in range(3):
    g = np.clip(rng.random((512, 512)).astype(np.float32), 0, 1)
    if trial == 2:
        g = gray_big
    new = guided_filter(g, g, radius=3, eps=1e-4)
    old = guided_filter_old(g, g, radius=3, eps=1e-4)
    all_ok &= report(f"guided self t{trial}", float(np.abs(new - old).max()), 2e-4)
g2 = np.clip(gray_big + rng.normal(0, 0.05, gray_big.shape).astype(np.float32), 0, 1)
all_ok &= report("guided cross (通用路径不变)",
                 float(np.abs(guided_filter(g2, gray_big) - guided_filter_old(g2, gray_big)).max()), 1e-6)

# 2. sauvola
for trial, win in [(0, None), (1, 31), (2, 75)]:
    t_n, m_n, s_n = sauvola_threshold(gray_big, window=win, k=0.24)
    t_o, m_o, s_o = sauvola_threshold_old(gray_big, window=win, k=0.24)
    all_ok &= report(f"sauvola t win={win}", float(np.abs(t_n - t_o).max()), 5e-4)
    all_ok &= report(f"sauvola m win={win}", float(np.abs(m_n - m_o).max()), 5e-4)
    all_ok &= report(f"sauvola s win={win}", float(np.abs(s_n - s_o).max()), 5e-4)

# 3. estimate_background 合并高斯
bg_n = estimate_background(gray_big)
bg_o = estimate_background_old(gray_big)
all_ok &= report("estimate_background", float(np.abs(bg_n - bg_o).max()), 5e-3)

# 4. normalize_illumination
for s in (0.3, 0.8, 1.0):
    all_ok &= report(f"normalize_illumination s={s}",
                     float(np.abs(normalize_illumination(gray_big, s)
                                  - normalize_illumination_old(gray_big, s)).max()), 5e-3)

# 5. verify_with_gradient + _otsu_hi
mag = np.abs(rng.normal(0.5, 0.2, gray_big.shape)).astype(np.float32)
ink = np.clip(rng.random(gray_big.shape).astype(np.float32) * mag, 0, 1)
all_ok &= report("verify_with_gradient",
                 float(np.abs(verify_with_gradient(ink, mag)
                              - verify_with_gradient_old(ink, mag)).max()), 1e-6)
hi_n, hi_o = _otsu_hi(ink), _otsu_hi_old(ink)
print(f"[{'PASS' if abs(hi_n - hi_o) < 3/255 else 'FAIL'}] _otsu_hi: {hi_o:.4f} -> {hi_n:.4f} (Δ={abs(hi_n-hi_o)*255:.1f}/255)")
all_ok &= abs(hi_n - hi_o) < 3 / 255

# 6. _is_pixel_art 颜色计数
for name in ["pixel_art.png", "sketch_photo.png", "blueprint.png", "photo_like.png", "document_scan.png"]:
    im = cv2.imread(f"/workspace/lineart-precise/demo/samples/{name}")
    im_big = cv2.resize(im, (2048, 1536), interpolation=cv2.INTER_NEAREST)
    c_n, c_o = _is_pixel_art(im_big), _is_pixel_art_old(im_big)
    ok = abs(c_n - c_o) < 1e-9
    print(f"[{'PASS' if ok else 'FAIL'}] _is_pixel_art {name}: old={c_o:.4f} new={c_n:.4f}")
    all_ok &= ok

# 7. detect_background：14 个样本 + 放大图
import glob
n_mismatch = 0
for p in sorted(glob.glob("/workspace/lineart-precise/demo/samples/*.png")):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for scale in (1.0, 2.0):
        im_t = im if scale == 1.0 else cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        mn = detect_background(im_t)
        mo = detect_background_old(im_t)
        if mn != mo:
            n_mismatch += 1
            print(f"  MISMATCH {p} x{scale}: {mo} -> {mn}")
print(f"[{'PASS' if n_mismatch == 0 else 'FAIL'}] detect_background: {28 - n_mismatch}/28 一致")
all_ok &= n_mismatch == 0

print("\n=== 总结:", "ALL PASS" if all_ok else "存在失败", "===")
sys.exit(0 if all_ok else 1)
