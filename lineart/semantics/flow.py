"""边缘切向流（ETF）与流线 FDoG。

算法源自 Kang et al. "Coherent Line Drawing" (NPAR 2007)，
参考实现：SSARCandy/Coherent-Line-Drawing (572★)、esimov/colidr (55★)、
uva-graphics/coherent_line_drawing (17★)。
本文件为全向量化 NumPy 重实现：
- ETF：切向流场迭代平滑（空间权 Ws + 梯度权 Wm + 方向一致权 Wd 与符号修正 Phi）
- flow_smooth：沿切向流线的 1D 高斯平滑，增强线条连贯性、抑制纹理
- fdog_semantic：流线平滑 + XDoG 的组合语义响应（只决定"哪里有线"，
  线条位置仍由梯度 NMS 决定，保证定位精度不降级）
"""
import math

import cv2
import numpy as np

from ..common import EPS, bilinear_sample, clip_f32
from .xdog import multiscale_xdog

# ---------------------------------------------------------------------------
# numba JIT 加速（可选依赖）：nopython 模式要求所有操作为数值计算，
# 不支持字符串操作；nogil=True 释放 GIL，cache=True 缓存编译结果。
# numba 不可用时回退到纯 NumPy，保持 backward compatibility。
# ---------------------------------------------------------------------------
try:
    import numba  # type: ignore

    _NUMBA_AVAILABLE = True
    _numba_jit = numba.jit(nopython=True, nogil=True, cache=True)
except ImportError:
    # numba 未安装，回退到纯 NumPy（无装饰器）
    _NUMBA_AVAILABLE = False
    _numba_jit = lambda fn: fn  # noqa: E731 – 用作透明回退装饰器


@_numba_jit
def _shift(a, dy, dx):
    """整数平移，出界填 0（避免 np.roll 的环绕污染）。

    性能：empty_like + 边缘清零替代 zeros_like 全图 memset
    （目标区会被源区覆盖，全图清零是纯浪费；240 次实测 -42%）。
    numba：nopython 模式编译为机器码，240 次调用实测 ~3x 加速。
    """
    out = np.empty_like(a)
    h, w = a.shape
    ys0, ys1 = max(0, dy), h + min(0, dy)
    yd0, yd1 = max(0, -dy), h + min(0, -dy)
    xs0, xs1 = max(0, dx), w + min(0, dx)
    xd0, xd1 = max(0, -dx), w + min(0, -dx)
    out[yd0:yd1, xd0:xd1] = a[ys0:ys1, xs0:xs1]
    if yd0 > 0:
        out[:yd0, :] = 0
    if yd1 < h:
        out[yd1:, :] = 0
    if xd0 > 0:
        out[:, :xd0] = 0
    if xd1 < w:
        out[:, xd1:] = 0
    return out


def _tanh(z):
    """np.tanh 直接调用（NumPy 2.x 对 float32 有 SIMD 实现，
    1.5M 元素实测 3.4ms vs 旧 LUT fancy-index 5.3ms，且无 4.9e-3
    量化误差——更快也更准，LUT 已废弃）。"""
    return np.tanh(z)


def compute_etf(gray, kernel=5, iters=4):
    """计算边缘切向流场。返回单位切向量 (tx, ty)，与梯度垂直。

    性能：tanh 用查表（NumPy tanh 是标量循环，24 邻居×5 迭代下是瓶颈）；
    valid 检查删除——tx/ty 初始化为单位向量（无效处置为 (1,0)），恒有效。
    注意：cv2.Sobel / cv2.magnitude / np.percentile 不支持 numba nopython，
    故本函数不做 JIT 装饰；_shift 子函数已 JIT 加速。
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    peak = float(np.percentile(mag, 99.9)) + EPS
    mag = np.clip(mag / peak, 0.0, 1.0)

    tx, ty = gy, -gx
    n = np.hypot(tx, ty)
    safe = n > 1e-8
    n = np.maximum(n, 1e-8)
    tx = np.where(safe, tx / n, 1.0).astype(np.float32)
    ty = np.where(safe, ty / n, 0.0).astype(np.float32)

    k2 = kernel // 2
    offsets, ws_list = [], []
    for dy in range(-k2, k2 + 1):
        for dx in range(-k2, k2 + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))
            ws_list.append(max(0.0, 1.0 - math.hypot(dy, dx) / kernel))

    # mag 在迭代中不变 → 24 个 (ws*wm) 权重场预计算一次，
    # 迭代内每邻居只剩 dot + 累加（内存约 24*N*4B，>768MB 时回退逐次计算）
    n_px = mag.size
    precompute = n_px * 24 * 4 <= 768 * 1024 * 1024
    if precompute:
        wwm_list = []
        for (dy, dx), ws in zip(offsets, ws_list):
            nmag = _shift(mag, dy, dx)
            wm = 0.5 * (1.0 + _tanh(3.0 * (nmag - mag)))
            wwm_list.append((ws * wm).astype(np.float32))
        offsets_ws = list(zip(offsets, wwm_list))

    for _ in range(iters):
        sx = np.zeros_like(tx)
        sy = np.zeros_like(ty)
        if precompute:
            for (dy, dx), wwm in offsets_ws:
                ntx = _shift(tx, dy, dx)
                nty = _shift(ty, dy, dx)
                # w*ntx = wwm*|dot|*sign(dot)*ntx = wwm*dot*ntx（|x|*sign(x)=x）
                t = wwm * (tx * ntx + ty * nty)
                sx += t * ntx
                sy += t * nty
        else:
            for (dy, dx), ws in zip(offsets, ws_list):
                ntx = _shift(tx, dy, dx)
                nty = _shift(ty, dy, dx)
                nmag = _shift(mag, dy, dx)
                wm = 0.5 * (1.0 + _tanh(3.0 * (nmag - mag)))
                t = (ws * wm) * (tx * ntx + ty * nty)
                sx += t * ntx
                sy += t * nty
        ln = np.hypot(sx, sy)
        upd = ln > 1e-8
        ln = np.maximum(ln, 1e-8)
        tx = np.where(upd, sx / ln, tx)
        ty = np.where(upd, sy / ln, ty)
    return tx, ty


def flow_smooth(field, tx, ty, sigma_m=3.0, length=6):
    """沿切向流线的 1D 高斯平滑：线条方向信息被保留并贯通，
    法向（跨线）结构不受影响，纹理噪点因方向不一致被平均掉。

    性能：cv2.remap（C++ 双线性，BORDER_REPLICATE）替代手写花式索引采样，
    数值等价（clip 边界 == replicate）。
    """
    h, w = field.shape
    ys_col = np.arange(h, dtype=np.float32)[:, None]
    xs_row = np.arange(w, dtype=np.float32)[None, :]
    svals = np.arange(-length, length + 1, dtype=np.float32)
    ws = np.exp(-(svals ** 2) / (2.0 * sigma_m * sigma_m + EPS)).astype(np.float32)
    ws /= ws.sum()
    out = np.zeros_like(field)
    field32 = field.astype(np.float32, copy=False)
    for s, wgt in zip(svals, ws):
        if s == 0.0:
            out += wgt * field32
            continue
        map_y = ty * s + ys_col  # 广播生成连续 (h,w)
        map_x = tx * s + xs_row
        out += wgt * cv2.remap(field32, map_x, map_y, cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    return clip_f32(out)


def fdog_semantic(gray, coherence=0.6, detail=0.6, sensitivity=0.5,
                   work_scale=None, multiscale=False):
    """流线连贯的语义墨迹图（决定哪里有线，不决定线在哪）。

    流程（Kang et al. 思想的精度友好变体）：
    1. ETF 切向流场（大图自动降采样计算，语义图无需全分辨率）
    2. 灰度沿流线平滑 -> 抑制纹理、贯通线条
    3. 平滑结果上做多尺度 XDoG -> 语义响应
    4. 响应再做一轮短流线平滑 -> 线条连贯

    Args:
        gray: 输入灰度图，范围 [0, 1] 或 [0, 255]，自动归一化到 [0, 1]。
        coherence: 线条连贯性系数，范围 [0, 1]。值越大线条越连贯。
        detail: 细节保留系数，范围 [0, 1]。值越大细线保留越多。
        sensitivity: 边缘敏感度系数，范围 [0, 1]。值越大检测越灵敏。
        work_scale: 工作尺度。None 时自动选择（0.25/0.5/1.0）。
        multiscale: 是否启用多尺度语义融合。默认 False。
            当为 True 时，使用 3 个尺度（0.25, 0.5, 1.0）的语义图加权融合：
            - 小尺度（0.25）权重 0.5：细线保留
            - 中尺度（0.5）权重 0.3：结构完整性
            - 大尺度（1.0）权重 0.2：全局结构
            融合公式：ink = 0.5*ink_025 + 0.3*ink_05 + 0.2*ink_10
            适用于需要保留极细线条的场景。

    Returns:
        语义墨迹图，范围 [0, 1]，float32 类型。

    Note:
        输出尺寸与输入一致，无论 work_scale 如何设置。
        multiscale 模式计算量约为单尺度的 3 倍，建议仅在需要时启用。
    """
    h, w = gray.shape
    if work_scale is None:
        # 语义层无需全分辨率（定位精度由梯度层保证），大图降采样计算。
        # v1.8.0: 0.35 -> 0.25（4096 图提速约 32%）；A/B 验证真实样本放大
        # 4096 后严格 IoU >= 0.9997、容差(3px) IoU = 1.0、差异全部为
        # 边缘带像素且无孤立连通域（无结构性丢线）。
        m = max(h, w)
        work_scale = 0.25 if m > 2560 else (0.5 if m > 1536 else 1.0)

    # 多尺度语义融合：3 个尺度加权融合，小尺度权重更高（细线保留）
    if multiscale:
        # 权重：小尺度 0.5（细线保留）> 中尺度 0.3（结构完整性）> 大尺度 0.2（全局结构）
        w1, w2, w3 = 0.5, 0.3, 0.2
        ink_025 = _compute_semantic(gray, coherence, detail, sensitivity, 0.25)
        ink_05 = _compute_semantic(gray, coherence, detail, sensitivity, 0.5)
        ink_10 = _compute_semantic(gray, coherence, detail, sensitivity, 1.0)
        ink = w1 * ink_025 + w2 * ink_05 + w3 * ink_10
        return np.clip(ink, 0.0, 1.0).astype(np.float32)

    if work_scale < 1.0:
        nw, nh = int(round(w * work_scale)), int(round(h * work_scale))
        work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        work = gray

    coh = float(np.clip(coherence, 0.0, 1.0))
    tx, ty = compute_etf(work, kernel=5, iters=3 + int(round(3 * coh)))

    sigma_m = 1.5 + 4.0 * coh
    length = 3 + int(round(6 * coh))
    smooth = flow_smooth(work, tx, ty, sigma_m=sigma_m, length=length)

    d = float(np.clip(detail, 0.0, 1.0))
    fine = 0.7 + (1.0 - d) * 0.9
    coarse = 1.5 + (1.0 - d) * 1.8
    eps = 0.030 - 0.024 * float(np.clip(sensitivity, 0.0, 1.0))
    ink = multiscale_xdog(smooth, sigmas=(fine, coarse), eps=eps)

    ink = flow_smooth(ink.astype(np.float32), tx, ty,
                      sigma_m=max(1.0, sigma_m * 0.5),
                      length=max(2, length // 2))

    if work_scale < 1.0:
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(ink, 0.0, 1.0).astype(np.float32)


def _compute_semantic(gray, coherence, detail, sensitivity, scale):
    """内部函数：在指定尺度上计算语义墨迹图。

    Args:
        gray: 输入灰度图。
        coherence: 连贯性系数。
        detail: 细节系数。
        sensitivity: 敏感度系数。
        scale: 工作尺度（0.25/0.5/1.0）。

    Returns:
        语义墨迹图，范围 [0, 1]，尺寸与输入一致。
    """
    h, w = gray.shape
    if scale < 1.0:
        nw, nh = int(round(w * scale)), int(round(h * scale))
        work = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        work = gray

    coh = float(np.clip(coherence, 0.0, 1.0))
    tx, ty = compute_etf(work, kernel=5, iters=3 + int(round(3 * coh)))

    sigma_m = 1.5 + 4.0 * coh
    length = 3 + int(round(6 * coh))
    smooth = flow_smooth(work, tx, ty, sigma_m=sigma_m, length=length)

    d = float(np.clip(detail, 0.0, 1.0))
    fine = 0.7 + (1.0 - d) * 0.9
    coarse = 1.5 + (1.0 - d) * 1.8
    eps = 0.030 - 0.024 * float(np.clip(sensitivity, 0.0, 1.0))
    ink = multiscale_xdog(smooth, sigmas=(fine, coarse), eps=eps)

    ink = flow_smooth(ink.astype(np.float32), tx, ty,
                      sigma_m=max(1.0, sigma_m * 0.5),
                      length=max(2, length // 2))

    if scale < 1.0:
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(ink, 0.0, 1.0).astype(np.float32)
