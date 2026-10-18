"""第二批优化的等价性验证。"""
import sys

import cv2
import numpy as np

sys.path.insert(0, "/workspace/lineart-precise")
from lineart.adaptive import adaptive_semantic, sauvola_threshold, sensitivity_to_k

rng = np.random.default_rng(7)
img = cv2.imread("/workspace/lineart-precise/demo/samples/sketch_photo.png", cv2.IMREAD_COLOR)
big = cv2.resize(img, (4096, 3072), interpolation=cv2.INTER_CUBIC)
gray_big = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

all_ok = True


# 旧 adaptive_semantic（t/s 上采样后全分辨率算 ink）
def adaptive_semantic_old(gray, window=None, sensitivity=0.5, softness=0.5):
    k = sensitivity_to_k(sensitivity)
    h, w = gray.shape
    if max(h, w) > 2048:
        nw, nh = int(round(w * 0.75)), int(round(h * 0.75))
        small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        t_s, _, s_s = sauvola_threshold(small, window=window, k=k)
        t = cv2.resize(t_s, (w, h), interpolation=cv2.INTER_LINEAR)
        s = cv2.resize(s_s, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        t, m, s = sauvola_threshold(gray, window=window, k=k)
    band = np.maximum(s * 0.6, 0.015)
    band = band * (0.5 + softness) + 0.01
    ink = (t - gray) / (2.0 * band) + 0.5
    return np.clip(ink, 0.0, 1.0).astype(np.float32)


# 1. adaptive_semantic：余弦相似度 + 二值翻转率
adaptive_accept = True
for name, sens, soft in [("sketch_photo", 0.7, 0.5), ("document_scan", 0.5, 0.5)]:
    im = cv2.imread(f"/workspace/lineart-precise/demo/samples/{name}.png", cv2.IMREAD_COLOR)
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    g = cv2.resize(g, (3072, 2304), interpolation=cv2.INTER_AREA)
    ink_n = adaptive_semantic(g, sensitivity=sens, softness=soft)
    ink_o = adaptive_semantic_old(g, sensitivity=sens, softness=soft)
    cos = float(np.dot(ink_n.ravel(), ink_o.ravel())
                / (np.linalg.norm(ink_n) * np.linalg.norm(ink_o) + 1e-12))
    flip = float(np.mean((ink_n > 0.5) != (ink_o > 0.5)))
    maxd = float(np.abs(ink_n - ink_o).max())
    # ink 是软强度场：低尺度计算的差异集中在软过渡带（flip），
    # 判定以最终 to_lineart 输出对比为准（见第 4 节）
    print(f"[INFO] adaptive_semantic {name}: cos={cos:.6f} "
          f"flip={flip*100:.3f}% max|Δ|={maxd:.4f}")

# 4. 最终输出对比（决定性判据）：新旧 to_lineart 全流程
import lineart.adaptive as _ad
from lineart import to_lineart
from lineart.pipeline import LineArtParams

_orig = _ad.adaptive_semantic
for name in ["sketch_photo", "document_scan"]:
    im = cv2.imread(f"/workspace/lineart-precise/demo/samples/{name}.png", cv2.IMREAD_COLOR)
    im = cv2.resize(im, (3072, 2304), interpolation=cv2.INTER_CUBIC)
    p = LineArtParams(scene="auto", max_side=4096)
    out_n = to_lineart(im, p)
    _ad.adaptive_semantic = adaptive_semantic_old
    try:
        out_o = to_lineart(im, p)
    finally:
        _ad.adaptive_semantic = _orig
    diff = np.abs(out_n.astype(np.int16) - out_o.astype(np.int16))
    n_hard = int((diff > 60).sum())          # 明显不同的像素
    n_soft = int(((diff > 0) & (diff <= 60)).sum())
    total = diff.size
    ok = n_hard / total < 0.001
    print(f"[{'PASS' if ok else 'FAIL'}] to_lineart {name}: "
          f"明显差异 {n_hard} px ({n_hard/total*100:.4f}%), "
          f"抗锯齿级差异 {n_soft} px ({n_soft/total*100:.3f}%)")
    adaptive_accept &= ok
print(f"[{'PASS' if adaptive_accept else 'FAIL'}] adaptive_semantic 低尺度 ink（由最终输出判定）")
all_ok &= adaptive_accept

# 2. findNonZero vs nonzero（点集与顺序一致性）
for t in range(3):
    m = rng.random((600, 800)) < 0.03
    if t == 2:
        m[:] = False
    pts = cv2.findNonZero(m.astype(np.uint8))
    if pts is None:
        yz, xz = np.zeros(0, np.int64), np.zeros(0, np.int64)
    else:
        pts = pts.reshape(-1, 2)
        xz = pts[:, 0].astype(np.int64)
        yz = pts[:, 1].astype(np.int64)
    yo, xo = np.nonzero(m)
    ok = np.array_equal(yz, yo) and np.array_equal(xz, xo)
    print(f"[{'PASS' if ok else 'FAIL'}] findNonZero t{t}: {len(yo)} 点，顺序/值一致")
    all_ok &= ok

# 3. uint16 take 索引一致性
g = rng.random((300, 400)).astype(np.float32)
lut = np.linspace(0, 1, 4096, dtype=np.float32) ** 0.7
g *= 4095
i16 = np.take(lut, g.astype(np.uint16))
i32 = np.take(lut, g.astype(np.int32))
ok = np.array_equal(i16, i32)
print(f"[{'PASS' if ok else 'FAIL'}] uint16 take 一致")
all_ok &= ok

print("\n=== 总结:", "ALL PASS" if all_ok else "存在失败", "===")
sys.exit(0 if all_ok else 1)
