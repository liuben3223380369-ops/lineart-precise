"""micro-bench：第 13 轮速率优化的方案对比实验。

1. bilinear_splat 的 np.add.at ×4 vs bincount vs 单次拼接 add.at
2. _shift 的 zeros_like vs empty_like+边缘清零
3. tanh：LUT vs np.tanh（float32 SIMD）
4. ETF 迭代：逐邻居 vs (24,H,W) 堆叠 reduce
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart.common import bilinear_splat

rng = np.random.default_rng(0)


def bench(label, fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(f"  {label:42s} {min(ts)*1000:8.1f} ms")
    return min(ts)


print("== 1. bilinear_splat（180K 点，4096 画布） ==")
H, W = 3072, 4096
n_pts = 180_000
py = rng.uniform(0, H - 1.01, n_pts).astype(np.float32)
px = rng.uniform(0, W - 1.01, n_pts).astype(np.float32)
vals = rng.uniform(0, 1, n_pts).astype(np.float32)


def splat_current():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat(canvas, py, px, vals)
    return canvas


def splat_addat1():
    canvas = np.zeros((H, W), dtype=np.float32)
    y0f, x0f = np.floor(py), np.floor(px)
    fy, fx = py - y0f, px - x0f
    y0 = y0f.astype(np.int64); x0 = x0f.astype(np.int64)
    y1 = y0 + 1; x1 = x0 + 1
    idx = np.concatenate([y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1])
    wgt = np.concatenate([vals * (1 - fy) * (1 - fx), vals * (1 - fy) * fx,
                          vals * fy * (1 - fx), vals * fy * fx])
    np.add.at(canvas.ravel(), idx, wgt)
    return canvas


def splat_bincount():
    canvas = np.zeros((H, W), dtype=np.float32)
    y0f, x0f = np.floor(py), np.floor(px)
    fy, fx = py - y0f, px - x0f
    y0 = y0f.astype(np.int64); x0 = x0f.astype(np.int64)
    y1 = y0 + 1; x1 = x0 + 1
    idx = np.concatenate([y0 * W + x0, y0 * W + x1, y1 * W + x0, y1 * W + x1])
    wgt = np.concatenate([vals * (1 - fy) * (1 - fx), vals * (1 - fy) * fx,
                          vals * fy * (1 - fx), vals * fy * fx]).astype(np.float64)
    acc = np.bincount(idx, weights=wgt, minlength=H * W)
    canvas += acc.reshape(H, W)
    return canvas


r0 = splat_current()
r1 = splat_addat1()
r2 = splat_bincount()
print(f"  等价性: addat1 max|Δ|={np.abs(r0-r1).max():.2e}  bincount max|Δ|={np.abs(r0-r2).max():.2e}")
bench("现状（4 次 np.add.at）", splat_current)
bench("单次拼接 np.add.at", splat_addat1)
bench("np.bincount", splat_bincount)

print("== 2. _shift（1434x1075 float32 ×240 次） ==")
h2, w2 = 1075, 1434
a = rng.uniform(0, 1, (h2, w2)).astype(np.float32)


def shift_current():
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            out = np.zeros_like(a)
            ys0, ys1 = max(0, dy), h2 + min(0, dy)
            yd0, yd1 = max(0, -dy), h2 + min(0, -dy)
            xs0, xs1 = max(0, dx), w2 + min(0, dx)
            xd0, xd1 = max(0, -dx), w2 + min(0, -dx)
            out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]


def shift_empty():
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            out = np.empty_like(a)
            ys0, ys1 = max(0, dy), h2 + min(0, dy)
            yd0, yd1 = max(0, -dy), h2 + min(0, -dy)
            xs0, xs1 = max(0, dx), w2 + min(0, dx)
            xd0, xd1 = max(0, -dx), w2 + min(0, -dx)
            out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
            # 边缘清零（4 条带，写入量 << 全图）
            if yd0 > 0: out[:yd0, :] = 0
            if yd1 < h2: out[yd1:, :] = 0
            if xd0 > 0: out[:, :xd0] = 0
            if xd1 < w2: out[:, xd1:] = 0


bench("现状（zeros_like memset）", shift_current)
bench("empty_like + 边缘清零", shift_empty)

print("== 3. tanh（1.5M float32 ×24 次） ==")
z = rng.uniform(-6, 6, (h2, w2)).astype(np.float32)
_TANH_LUT = np.tanh(np.linspace(-3.0, 3.0, 2047, dtype=np.float32))
_TANH_SCALE = 2046.0 / 6.0


def tanh_lut():
    idx = ((z + 3.0) * _TANH_SCALE).astype(np.int32)
    return _TANH_LUT[np.clip(idx, 0, 2046)]


def tanh_native():
    return np.tanh(z)


def tanh_sigmoid_exp():
    # 0.5*(1+tanh(x)) = sigmoid(2x)；用 exp 实现
    e = np.exp(-2.0 * z)
    return 1.0 / (1.0 + e)


r_l, r_n = tanh_lut(), tanh_native()
print(f"  LUT vs 原生 max|Δ|={np.abs(r_l-r_n).max():.2e}")
bench("现状（LUT fancy index）", tanh_lut)
bench("np.tanh 原生", tanh_native)
bench("exp sigmoid 近似", tanh_sigmoid_exp)

print("== 4. ETF 迭代内核（24 邻居 × 5 迭代） ==")
wwm = rng.uniform(0, 1, (24, h2, w2)).astype(np.float32)
tx = rng.uniform(-1, 1, (h2, w2)).astype(np.float32)
ty = rng.uniform(-1, 1, (h2, w2)).astype(np.float32)
offs = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy or dx]


def _shift(a, dy, dx):
    out = np.zeros_like(a)
    hh, ww = a.shape
    ys0, ys1 = max(0, dy), hh + min(0, dy)
    yd0, yd1 = max(0, -dy), hh + min(0, -dy)
    xs0, xs1 = max(0, dx), ww + min(0, dx)
    xd0, xd1 = max(0, -dx), ww + min(0, -dx)
    out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    return out


def etf_current():
    tx_, ty_ = tx.copy(), ty.copy()
    for _ in range(5):
        sx = np.zeros_like(tx_); sy = np.zeros_like(ty_)
        for k, (dy, dx) in enumerate(offs):
            ntx = _shift(tx_, dy, dx)
            nty = _shift(ty_, dy, dx)
            t = wwm[k] * (tx_ * ntx + ty_ * nty)
            sx += t * ntx
            sy += t * nty
        ln = np.hypot(sx, sy)
        ln = np.maximum(ln, 1e-8)
        tx_ = np.where(ln > 1e-8, sx / ln, tx_)
        ty_ = np.where(ln > 1e-8, sy / ln, ty_)


def etf_stack():
    tx_, ty_ = tx.copy(), ty.copy()
    for _ in range(5):
        NX = np.stack([_shift(tx_, dy, dx) for dy, dx in offs])
        NY = np.stack([_shift(ty_, dy, dx) for dy, dx in offs])
        dot = NX * tx_ + NY * ty_          # (24,h,w) 广播
        t = wwm * dot
        sx = (t * NX).sum(axis=0)
        sy = (t * NY).sum(axis=0)
        ln = np.hypot(sx, sy)
        ln = np.maximum(ln, 1e-8)
        tx_ = np.where(ln > 1e-8, sx / ln, tx_)
        ty_ = np.where(ln > 1e-8, sy / ln, ty_)


bench("现状（逐邻居 shift+乘加）", etf_current)
bench("堆叠 (24,h,w) reduce", etf_stack)
