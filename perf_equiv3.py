"""第三批优化（estimate_bg 0.125 / render 原地 / 两步缩放）等价性验证。"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace/lineart-precise")
from lineart.illumination import normalize_illumination
from lineart.scenes import detect_scene
from lineart.pipeline import LineArtParams, to_lineart

all_ok = True
img = cv2.imread("/workspace/lineart-precise/demo/samples/sketch_photo.png", cv2.IMREAD_COLOR)
big = cv2.resize(img, (4096, 3072), interpolation=cv2.INTER_CUBIC)
gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


def estimate_background_old(gray, downscale=0.25, sigma_ratio=0.035):
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, int(round(w * downscale))),
                              max(1, int(round(h * downscale)))),
                       interpolation=cv2.INTER_AREA)
    sigma = max(1.0, max(small.shape) * sigma_ratio) * np.sqrt(2.0)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigma)
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


# 1. normalize_illumination 0.125 vs 0.25：场差异 + 最终输出
for s in (0.3, 0.8):
    n = normalize_illumination(gray_big, s)
    o = normalize_illumination_old(gray_big, s)
    cos = float(np.dot(n.ravel(), o.ravel()) / (np.linalg.norm(n) * np.linalg.norm(o) + 1e-12))
    maxd = float(np.abs(n - o).max())
    ok = cos > 0.9999 and maxd < 0.05
    print(f"[{'PASS' if ok else 'FAIL'}] illumination 0.125 s={s}: cos={cos:.6f} max|Δ|={maxd:.4f}")
    all_ok &= ok

# 最终输出（决定性）
import lineart.illumination as _il
_orig = _il.normalize_illumination
for name in ["sketch_photo", "document_scan", "photo_like"]:
    im = cv2.imread(f"/workspace/lineart-precise/demo/samples/{name}.png", cv2.IMREAD_COLOR)
    im = cv2.resize(im, (3072, 2304), interpolation=cv2.INTER_CUBIC)
    p = LineArtParams(scene="auto", max_side=4096)
    out_n = to_lineart(im, p)
    _il.normalize_illumination = normalize_illumination_old
    try:
        out_o = to_lineart(im, p)
    finally:
        _il.normalize_illumination = _orig
    diff = np.abs(out_n.astype(np.int16) - out_o.astype(np.int16))
    n_hard = int((diff > 60).sum())
    n_soft = int(((diff > 0) & (diff <= 60)).sum())
    total = diff.size
    ok = n_hard / total < 0.002
    print(f"[{'PASS' if ok else 'FAIL'}] to_lineart {name}: 硬差异 {n_hard} ({n_hard/total*100:.4f}%), "
          f"软差异 {n_soft} ({n_soft/total*100:.3f}%)")
    all_ok &= ok

# 2. detect_scene 两步缩放：场景标签一致性
import glob
for p in sorted(glob.glob("/workspace/lineart-precise/demo/samples/*.png")):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    s_n = detect_scene(im).scene
    s_up = detect_scene(cv2.resize(im, (4096, 3072), interpolation=cv2.INTER_AREA)).scene
    ok = s_n == s_up
    print(f"[{'PASS' if ok else 'FAIL'}] detect_scene {p.split('/')[-1]}: 原图={s_n} 4096图={s_up}")
    all_ok &= ok

print("\n=== 总结:", "ALL PASS" if all_ok else "存在失败", "===")
sys.exit(0 if all_ok else 1)
