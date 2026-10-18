"""抗锯齿渲染：亚像素坐标 → 墨迹画布。

- refine_and_render: 精化 + 渲染总编排（方法分发 + 共识滤波 +
  边缘对合并 + SSR）
- _render_splat: 双线性 splat，粗线沿法向对称展开
- SSR 超采样渲染（render_ss>1 且粗线）：2x 画布 splat 后
  INTER_AREA 下采样，线宽与边缘抗锯齿比 1x 点列展开更准确
"""
import cv2
import numpy as np

from ..common import bilinear_sample, bilinear_splat, bilinear_splat_batch, clip_f32
from .merge import pair_edge_mask
from .refine import (
    _consensus_delta,
    _refine_gaussian_vectorized,
    _refine_parabolic,
    _refine_zernike,
)


def refine_and_render(fit_field, mask, gx, gy, darkness=None,
                      thickness=1.0, step=0.75, method="parabolic",
                      render_ss=1, consensus=True,
                      merge_edges=False, merge_gap=6, dark_field=None,
                      adaptive_width=True):
    """亚像素精化 + 抗锯齿渲染。

    Args:
        fit_field: 拟合场（梯度幅值或强度场），[0,1] float32
        mask: 脊线 bool 掩码
        gx, gy: 梯度场
        darkness: 墨迹强度（None 则用 fit_field）
        thickness: 线条粗细
        step: 采样步长（parabolic/gaussian）
        method: "parabolic" | "gaussian" | "zernike"
        render_ss: 渲染超采样倍率（2 = 2x 画布 splat 后 INTER_AREA
            下采样，细线的面积覆盖抗锯齿优于单点双线性 splat）
        consensus: 对亚像素 offset 做邻域一致性滤波（剔除拟合离群）
        merge_edges: 粗线双边缘合并为线中心（配对点的亚像素精化
            位置取平均；细线/单边缘无对面点，不受影响）
        merge_gap: 边缘对合并的最大搜索距离（像素）
        dark_field: [0,1] 亮度场（暗谷判据用，防止近距平行线误合并）

    Returns:
        canvas: [0,1] float32 抗锯齿墨迹图（1=墨）
    """
    h, w = fit_field.shape

    if method == "parabolic":
        ys, xs, dy, dx, delta = _refine_parabolic(fit_field, mask, gx, gy, step)
    elif method == "gaussian":
        ys, xs, dy, dx, delta = _refine_gaussian_vectorized(fit_field, mask, gx, gy, step)
    elif method == "zernike":
        ys, xs, dy, dx, delta = _refine_zernike(fit_field, mask, gx, gy)
    else:
        raise ValueError(f"未知精化方法: {method}")

    if len(ys) == 0:
        return np.zeros((h, w), dtype=np.float32)

    if consensus and method != "zernike":
        delta = _consensus_delta(ys, xs, dy, dx, delta, (h, w))

    py = ys + delta * dy
    px = xs + delta * dx

    if darkness is None:
        vals = clip_f32(fit_field[ys, xs])
    else:
        vals = clip_f32(darkness[ys, xs])

    if merge_edges and len(ys) > 1:
        # 边缘对合并：粗线的两侧脊线 → 线心（亚像素中点），
        # 恢复"一根线画一根线"的结构保真（v1.7.0）。
        # 中点继承配对间距为线宽：沿法向展开渲染，粗线仍是粗线
        # （不展开会让 3px 线变 1px 细线，总墨减半）。
        # 中点按半像素网格去重（多层脊线产生重复中点），
        # 未参与任何配对的点保持原位（真实单边缘）
        ia, ib = pair_edge_mask(mask, gx, gy, dark_field=dark_field,
                                max_gap=merge_gap)
        if len(ia):
            # 线宽估计 = 配对间距 - 1（脊线各向外偏 ~1px）
            mw = np.hypot(py[ia] - py[ib], px[ia] - px[ib]) - 1.0
            kk = np.round(mw).astype(np.int64)
            # 间距 2 的配对是细线双脊线（伴随线 splat 已重叠融合，
            # 合并只会减墨）——只合并间距 >=3 的真粗线（伴随线分离）
            keep_pair = kk >= 2
            ia = ia[keep_pair]
            ib = ib[keep_pair]
            kk = kk[keep_pair]
        if len(ia):
            my = (py[ia] + py[ib]) * 0.5
            mx = (px[ia] + px[ib]) * 0.5
            # 线浓度取两脊线侧的最大值（粗线渲染展开点各承载线浓度）
            mv = np.maximum(vals[ia], vals[ib])
            # 展开法向取 ia 端方向：两端脊线法向相反（dot<-0.5 才配对），
            # 取平均会得零向量 → 展开点全部重叠、墨被 clip 浪费。
            # 对称展开对法向符号不敏感，ia 端方向即线的真实法向
            mdy = dy[ia].copy()
            mdx = dx[ia].copy()
            key = np.round(my * 2.0).astype(np.int64) * (2 * w + 4) \
                + np.round(mx * 2.0).astype(np.int64)
            # 同一半像素网格的多对中点：位置取组内平均（亚像素信息
            # 不丢弃，且与输入顺序无关；组首会引入顺序偏差），
            # 浓度取组内最浓、线宽取组内最大（粗一点优于断线）。
            # 法向仍取组首：组内 ia 端法向理论同向，取首可避免
            # 极端情况下组内法向冲突相消。
            order = np.argsort(key, kind="stable")
            sk = key[order]
            first = np.concatenate(([True], sk[1:] != sk[:-1]))
            starts = np.where(first)[0]
            counts = np.diff(np.append(starts, len(sk))).astype(np.float32)
            my = np.add.reduceat(my[order], starts) / counts
            mx = np.add.reduceat(mx[order], starts) / counts
            mdy = mdy[order][starts]
            mdx = mdx[order][starts]
            mv = np.maximum.reduceat(mv[order], starts)
            kk = np.maximum.reduceat(kk[order], starts)
            # FWHM 线宽测量：从亮度场直接测（硬边线脊线间距≈宽，
            # 平滑细线间距≈宽+1，间距推导不可靠；测量值物理正确）
            if dark_field is not None:
                mw_f = _fwhm_widths(dark_field, my, mx, mdy, mdx,
                                    gradient=fit_field, adaptive=adaptive_width)
                kk = np.maximum(1, np.round(mw_f).astype(np.int64))
            # 沿法向（垂直脊线切向）展开：线宽 kk 的线用 kk 个点
            if kk.max() > 1:
                sel = np.repeat(np.arange(len(my)), kk)
                heads = np.concatenate(([0], np.cumsum(kk)[:-1]))
                j = np.arange(len(sel)) - heads[sel]
                off = (j - (kk[sel] - 1) / 2.0).astype(np.float32)
                ex, ey = -mdx[sel], -mdy[sel]  # 法向分量
                my = my[sel] + ey * off
                mx = mx[sel] + ex * off
                mv = mv[sel]
                mdy = mdy[sel]
                mdx = mdx[sel]
            un = np.ones(len(py), dtype=bool)
            un[ia] = False
            un[ib] = False
            py = np.concatenate((my, py[un]))
            px = np.concatenate((mx, px[un]))
            vals = np.concatenate((mv, vals[un]))
            dy = np.concatenate((mdy, dy[un]))
            dx = np.concatenate((mdx, dx[un]))

    ss = max(1, int(render_ss))
    if ss > 1 and thickness > 1.0:
        # 超采样渲染（仅粗线）：2x 网格上笔触展开 + INTER_AREA 下采样，
        # 粗线宽度与边缘抗锯齿比 1x 点列展开更准确。
        # 细线（thickness≤1）不适用：单点 splat 在 SSR 网格上的支撑只有
        # 1 个输出像素宽，AREA 平均会把双线性的亚像素分布量化到
        # 半像素网格（实测定位误差 0.063→0.169px），故细线走 1x 路径。
        canvas = np.zeros((h * ss, w * ss), dtype=np.float32)
        _render_splat(canvas, py * ss, px * ss, vals * (ss * ss),
                      thickness * ss, dy, dx)
        canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_AREA)
    else:
        canvas = np.zeros((h, w), dtype=np.float32)
        _render_splat(canvas, py, px, vals, thickness, dy, dx)
    return canvas


def _fwhm_widths(field, my, mx, ny, nx, dmax=2.5, gradient=None, adaptive=True):
    """沿法向测合并线的半高宽（FWHM）线宽估计。

    脊线间距与真实线宽的关系随线软硬变化（硬边线脊线在边界上
    间距≈宽；平滑细线脊线外偏间距≈宽+1），不可从间距推导。
    直接从亮度场测：线心浓度 c、远端背景 bg（两侧最大值），
    阈值 c+0.5(bg-c)，两侧各找首个过阈距离，线宽 = 两半之和。

    Args:
        gradient: 梯度幅值（用于自适应阈值系数选择），None 时使用默认系数
        adaptive: 是否启用自适应线宽（根据线类型调整阈值系数）
            - 硬边线（gradient >= 0.5 * max_gradient）：系数 0.4（更窄）
            - 软线（gradient < 0.5 * max_gradient）：系数 0.6（更宽）
    """
    ds = np.array([0.5, 1.0, 1.5, 2.0, 2.5], dtype=np.float32)
    bp = bilinear_sample(field, my + ny * dmax, mx + nx * dmax)
    bm = bilinear_sample(field, my - ny * dmax, mx - nx * dmax)
    bg = np.maximum(np.maximum(bp, bm), 0.0)
    c = bilinear_sample(field, my, mx)

    # 自适应阈值系数：硬边线更窄（0.4），软线更宽（0.6）
    if adaptive and gradient is not None:
        max_grad = float(gradient.max())
        grad_at_center = bilinear_sample(gradient, my, mx)
        # 分界点：0.5 * max_gradient
        is_hard_edge = grad_at_center >= 0.5 * max_grad
        k = np.where(is_hard_edge, np.float32(0.4), np.float32(0.6))
    else:
        k = np.float32(0.5)
    thr = c + k * (bg - c)
    thr = np.maximum(thr, c)  # bg<c（出界/异常）时退化为"恒不过阈"

    def _half(sgn):
        vals = np.stack([bilinear_sample(field, my + sgn * ny * d,
                                         mx + sgn * nx * d)
                         for d in ds], axis=1)
        over = vals >= thr[:, None]
        any_over = over.any(axis=1)
        first = np.argmax(over, axis=1)
        # 档间线性插值：前一档值与当前档值之间找交点
        i0 = np.clip(first - 1, 0, len(ds) - 1)
        v0 = np.take_along_axis(vals, i0[:, None], 1)[:, 0]
        v1 = np.take_along_axis(vals, first[:, None], 1)[:, 0]
        d0 = ds[i0]
        d1 = ds[first]
        frac = np.clip((thr - v0) / np.maximum(np.abs(v1 - v0), 1e-6),
                       0.0, 1.0)
        d_cross = d0 + frac * (d1 - d0)
        return np.where(any_over, np.minimum(d_cross, dmax), dmax)

    return _half(1.0) + _half(-1.0)


def _render_splat(canvas, py, px, vals, thickness, dy, dx):
    """双线性 splat 渲染，支持粗线对称展开（批量化）。"""
    if thickness <= 1.0:
        bilinear_splat_batch(canvas, py, px, vals)
    else:
        nx = -dy
        ny = dx
        n_pts = max(2, int(np.ceil(thickness - 1.0)) + 1)
        span = thickness - 1.0
        offsets = np.linspace(-span / 2.0, span / 2.0, n_pts)
        # 批量收集所有展开点的坐标与值，单次 bincount 累加
        n = len(py)
        k = len(offsets)
        total = n * k
        # 预分配批量数组：每个原始点展开 k 个偏移点
        py_batch = np.empty(total, dtype=py.dtype)
        px_batch = np.empty(total, dtype=px.dtype)
        vals_batch = np.empty(total, dtype=vals.dtype)
        for i, off in enumerate(offsets):
            sl = slice(i * n, (i + 1) * n)
            py_batch[sl] = py + ny * off
            px_batch[sl] = px + nx * off
            vals_batch[sl] = vals
        bilinear_splat_batch(canvas, py_batch, px_batch, vals_batch)
