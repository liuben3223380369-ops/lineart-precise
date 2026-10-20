"""亚像素边缘精化：把线条中心定位到亚像素精度。

支持三种算法：
- parabolic: 三点抛物线拟合（基线，最快）
- gaussian: 五点高斯拟合（更精确，边缘剖面近似高斯）
- zernike: Zernike 矩精化（超精度，0.01px 级）

另含 _consensus_delta：亚像素偏移的邻域中值共识滤波
（Stable Edge Region 思想，arXiv:2502.16502），剔除孤立拟合噪声。
"""
import math

import cv2
import numpy as np

from ..common import EPS, bilinear_sample


# ─── 抛物线拟合 ──────────────────────────────────────────────────

def _refine_parabolic(fit_field, mask, gx, gy, step):
    """三点抛物线拟合：向量化实现。"""
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ys, xs, np.zeros(len(ys), dtype=np.float32), np.zeros(len(ys), dtype=np.float32)

    gxx = gx[ys, xs]
    gyy = gy[ys, xs]
    mag = np.hypot(gxx, gyy)
    safe = mag > EPS
    dx = np.zeros_like(gxx)
    dy = np.zeros_like(gyy)
    dx[safe] = gxx[safe] / mag[safe]
    dy[safe] = gyy[safe] / mag[safe]

    s0 = fit_field[ys, xs]
    sm = bilinear_sample(fit_field, ys - dy * step, xs - dx * step)
    sp = bilinear_sample(fit_field, ys + dy * step, xs + dx * step)
    denom = sm - 2.0 * s0 + sp
    delta = np.zeros_like(s0)
    nz = np.abs(denom) > EPS
    delta[nz] = np.clip((sm[nz] - sp[nz]) / (2.0 * denom[nz]),
                        -0.5, 0.5) * step
    return ys, xs, dy, dx, delta


# ─── 高斯拟合 ────────────────────────────────────────────────────

def gaussian_fit_subpixel(values, step=0.5):
    """对 1D 强度剖面拟合高斯函数，返回亚像素偏移。

    values: 5 个采样点 [f(-2s), f(-s), f(0), f(+s), f(+2s)]
    step: 采样步长

    原理：高斯取 log 后变抛物线，用加权最小二乘拟合 log(f) = a*x^2 + b*x + c
    边缘位置 mu = b / (2*|a|)
    """
    s = float(step)
    xs = np.array([-2, -1, 0, 1, 2], dtype=np.float64) * s
    vals = np.asarray(values, dtype=np.float64)
    vmax = vals.max()
    if vmax <= 0:
        return 0.0
    vals = vals / vmax
    logv = np.log(np.maximum(vals, 1e-10))

    w = np.array([0.2, 0.5, 1.0, 0.5, 0.2], dtype=np.float64)
    W = np.diag(w)
    X = np.column_stack([xs ** 2, xs, np.ones(5)])
    try:
        coef, _, _, _ = np.linalg.lstsq(W @ X, W @ logv, rcond=None)
    except Exception:
        return 0.0
    a, b, c = coef
    if a >= -1e-8:
        return 0.0
    mu = b / (2.0 * abs(a))
    return float(np.clip(mu, -1.5 * s, 1.5 * s))


def _refine_gaussian_vectorized(fit_field, mask, gx, gy, step=0.5):
    """高斯拟合亚像素精化：向量化实现（比逐像素快 100x）。"""
    # cv2.findNonZero（C++ 行优先扫描）替代 np.nonzero，12M 像素 69ms → ~20ms
    # OpenCV 5.x 返回 (N,2) 的 [x,y]，4.x 返回 (N,1,2)，reshape 统一处理
    pts = cv2.findNonZero(mask.astype(np.uint8))
    if pts is None:
        e = np.zeros(0, dtype=np.float32)
        return (e.astype(np.int64), e.astype(np.int64), e, e, e)
    pts = pts.reshape(-1, 2)
    xs = pts[:, 0].astype(np.int64)
    ys = pts[:, 1].astype(np.int64)

    gxx = gx[ys, xs]
    gyy = gy[ys, xs]
    mag = np.hypot(gxx, gyy)
    safe = mag > EPS
    dx = np.zeros_like(gxx)
    dy = np.zeros_like(gyy)
    dx[safe] = gxx[safe] / mag[safe]
    dy[safe] = gyy[safe] / mag[safe]

    # 5 点采样
    svals = np.array([-2, -1, 0, 1, 2], dtype=np.float32) * step
    samples = np.zeros((len(ys), 5), dtype=np.float32)
    for i, s in enumerate(svals):
        samples[:, i] = bilinear_sample(fit_field, ys + dy * s, xs + dx * s)

    # 向量化高斯拟合
    vmax = samples.max(axis=1, keepdims=True)
    vmax = np.maximum(vmax, EPS)
    norm = samples / vmax
    logv = np.log(np.maximum(norm, 1e-10))

    # 加权最小二乘：log(f) = a*x^2 + b*x + c
    w = np.array([0.2, 0.5, 1.0, 0.5, 0.2], dtype=np.float64)
    xs_arr = np.array([-2, -1, 0, 1, 2], dtype=np.float64) * step

    # 预计算 X^T W X（对称正定，逐点相同）与 X^T W y（逐点不同）
    wxs2 = w * xs_arr ** 2
    wxs = w * xs_arr
    wxs4 = w * xs_arr ** 4
    wxs3 = w * xs_arr ** 3

    Sxx = wxs4.sum()
    Sxy = wxs3.sum()
    Sxz = wxs2.sum()
    Syy = wxs2.sum()
    Syz = wxs.sum()
    Szz = w.sum()

    b1 = (wxs2 * logv).sum(axis=1)
    b2 = (wxs * logv).sum(axis=1)
    b3 = (w * logv).sum(axis=1)

    # 克莱姆法则解 3x3 系统（矩阵固定 → det 预计算一次，全向量化）：
    # A = [[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]]
    det = (Sxx * (Syy * Szz - Syz * Syz)
           - Sxy * (Sxy * Szz - Syz * Sxz)
           + Sxz * (Sxy * Syz - Syy * Sxz))
    delta = np.zeros(len(ys), dtype=np.float32)
    if abs(det) > 1e-12:
        inv_det = 1.0 / det
        # a：第 1 列替换为 b
        det_a = (b1 * (Syy * Szz - Syz * Syz)
                 - Sxy * (b2 * Szz - Syz * b3)
                 + Sxz * (b2 * Syz - Syy * b3))
        # b：第 2 列替换为 b
        det_b = (Sxx * (b2 * Szz - Syz * b3)
                 - b1 * (Sxy * Szz - Syz * Sxz)
                 + Sxz * (Sxy * b3 - b2 * Sxz))
        a = det_a * inv_det
        bcoef = det_b * inv_det
        ok = a < -1e-8
        mu = np.where(ok, bcoef / (2.0 * np.abs(np.where(ok, a, 1.0))), 0.0)
        delta = np.clip(mu, -1.5 * step, 1.5 * step).astype(np.float32)

    return ys, xs, dy, dx, delta


# ─── Zernike 矩超精度模式 ────────────────────────────────────────

def _zernike_kernel(n, m, size):
    """生成 Zernike 多项式 (n,m) 在 size×size 网格上的核。"""
    yy, xx = np.mgrid[-1:1:1j*size, -1:1:1j*size]
    r = np.hypot(yy, xx)
    theta = np.arctan2(yy, xx)
    mask = (r <= 1.0).astype(np.float32)

    R = np.zeros((size, size), dtype=np.float32)
    for k in range(0, (n - abs(m)) // 2 + 1):
        num = math.factorial(n - k)
        den = (math.factorial(k) *
               math.factorial((n + abs(m)) // 2 - k) *
               math.factorial((n - abs(m)) // 2 - k))
        coef = num / den if den > 0 else 0
        R += coef * (r ** (n - 2 * k))

    if m >= 0:
        Z = R * np.cos(m * theta)
    else:
        Z = R * np.sin(-m * theta)
    return Z * mask


def zernike_subpixel_edge(patch, order=2):
    """Zernike 矩亚像素边缘检测。"""
    h, w = patch.shape
    size = min(h, w)

    patch_n = patch.astype(np.float32)
    pmax, pmin = patch_n.max(), patch_n.min()
    if pmax - pmin < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    patch_n = (patch_n - pmin) / (pmax - pmin)

    Z00 = _zernike_kernel(0, 0, size)
    Z11r = _zernike_kernel(1, 1, size)
    Z11i = _zernike_kernel(1, -1, size)
    Z20 = _zernike_kernel(2, 0, size)

    M00 = np.sum(patch_n * Z00)
    M11r = np.sum(patch_n * Z11r)
    M11i = np.sum(patch_n * Z11i)
    M20 = np.sum(patch_n * Z20)

    theta = np.arctan2(M11i, M11r)
    delta_n = (3.0 * M20) / (2.0 * M00 + 1e-10) if abs(M00) > 1e-10 else 0.0
    strength = np.hypot(M11r, M11i) / (np.pi * size * size / 4.0 + 1e-10)
    delta_n = float(np.clip(delta_n, -0.8, 0.8))

    return delta_n, 0.0, float(theta), float(strength)


def _refine_zernike(fit_field, mask, gx, gy, patch_size=7):
    """Zernike 矩精化（适合直线边缘）。

    性能：逐像素 7×7 矩计算（Python for 循环），大图较慢。
    非默认方法（默认 gaussian 向量化实现），仅精度极限场景按需启用。
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return ys, xs, np.zeros(len(ys), dtype=np.float32), np.zeros(len(ys), dtype=np.float32), np.zeros(len(ys), dtype=np.float32)

    half = patch_size // 2
    pad = np.pad(fit_field, half, mode='symmetric')

    gxx = gx[ys, xs]
    gyy = gy[ys, xs]
    mag = np.hypot(gxx, gyy)
    safe = mag > EPS
    dx = np.zeros_like(gxx)
    dy = np.zeros_like(gyy)
    dx[safe] = gxx[safe] / mag[safe]
    dy[safe] = gyy[safe] / mag[safe]

    deltas = np.zeros(len(ys), dtype=np.float32)
    for i, (y, x) in enumerate(zip(ys, xs)):
        patch = pad[y:y + patch_size, x:x + patch_size]
        dn, _, theta, _ = zernike_subpixel_edge(patch, order=2)
        grad_dir = np.arctan2(gy[y, x], gx[y, x])
        angle_diff = theta - grad_dir
        deltas[i] = dn * np.cos(angle_diff)

    return ys, xs, dy, dx, deltas


# ─── 偏移一致性滤波 ──────────────────────────────────────────────

def _consensus_delta(ys, xs, dy, dx, delta, shape):
    """offset 一致性滤波（Stable Edge Region 思想，arXiv:2502.16502）。

    同一条线的亚像素偏移沿脊线缓变；孤立点的大偏移多为拟合噪声。
    取每个脊点 3x3 邻域内（有线点）delta 的中值作为局部共识
    （中值对离群鲁棒，优于均值），偏离共识超过 0.2px 的点直接
    替换为共识值（拟合噪声的单点修正，连续渐变的真实偏移不受影响）。
    """
    if len(ys) < 9:
        return delta
    h, w = shape
    grid = np.zeros((h, w), dtype=np.float32)
    cnt = np.zeros((h, w), dtype=np.float32)
    np.add.at(grid, (ys, xs), delta)
    np.add.at(cnt, (ys, xs), 1.0)

    n = len(ys)
    d9 = np.full((9, n), np.nan, dtype=np.float32)
    i = 0
    for oy in (-1, 0, 1):
        for ox in (-1, 0, 1):
            yy = np.clip(ys + oy, 0, h - 1)
            xx = np.clip(xs + ox, 0, w - 1)
            c = cnt[yy, xx]
            has = c > 0
            d9[i] = np.where(has, grid[yy, xx] / np.maximum(c, 1.0), np.nan)
            i += 1
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(d9, axis=0)
    # 无任何有线邻居的孤立点（med=nan）保持原值
    med = np.where(np.isnan(med), delta, med)
    dev = delta - med
    # 软加权过渡（v1.20.0）：硬阈值 0.2 在边界处有不连续跳变，
    # 同一条线上相邻点的 dev 若分别为 0.19/0.21 会一个保留一个替换，
    # 产生亚像素抖动。软过渡带 [0.15, 0.25]：smoothstep 混合，
    # 偏离小则多保留原值、偏离大则多取共识，消除跳变。
    abs_dev = np.abs(dev)
    w = np.clip((abs_dev - 0.15) * 10.0, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)  # smoothstep
    return (delta * (1.0 - w) + med * w).astype(np.float32)
