"""边缘对合并：粗线的双边缘脊线 → 线中心（结构保真）。

边缘检测本质（Sobel/NMS 梯度脊线）对线宽 >=2px 的线给出两条脊线
（各在线的一侧、梯度方向指向亮侧），直接渲染会把一根线画成两根
伴随线（实测 3px 硬边线：core 丢线 33%、伴随虚假墨 1463px）。

本模块在整数 mask 上收集「间距 <= max_gap、法向相反、中点存在
暗谷」的脊线点对（允许一个点参与多对——脊线常为 2px 宽，两层
各自需要中点，否则外层以原位渲染产生伴随线），渲染层把点对的
亚像素精化位置取平均得到线心，并按半像素网格去重。暗谷判据区分
"一根粗线的两边缘"（中间暗）与"两根近距细线"（中间是亮背景），
防止排线/纹理被误合并。
"""
import numpy as np

from ..common import bilinear_sample


def pair_edge_mask(mask, gx, gy, dark_field=None, max_gap=6,
                   dot_min=-0.5, dark_ratio=0.75):
    """整数 mask 上收集边缘脊线点对（沿梯度负方向搜索对面脊线）。

    Args:
        mask: 脊线 bool 掩码
        gx, gy: 梯度场（法向来源；梯度指向亮侧，线心在 -n 方向）
        dark_field: [0,1] 亮度场（可选）。提供时要求配对中点亮度
            < dark_ratio * min(两侧亮部亮度)，否则不合并；
            亮部采样越界返回 0 → 判据恒假 → 保守不合并
        max_gap: 最大搜索距离（像素），覆盖线宽 ~max_gap+1
        dot_min: 对面点法向点积上限（负值 = 方向相反）

    Returns:
        (ia, ib): 点对索引数组（ia < ib，每对一次；同一个点可以
            出现在多对中——多层脊线各自产生中点），无配对时为空
    """
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n < 2:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    h, w = mask.shape
    gm = np.maximum(np.hypot(gx[ys, xs], gy[ys, xs]), 1e-12)
    nx = (gx[ys, xs] / gm).astype(np.float32)
    ny = (gy[ys, xs] / gm).astype(np.float32)

    idx_map = np.full((h, w), -1, dtype=np.int32)
    idx_map[ys, xs] = np.arange(n, dtype=np.int32)

    if dark_field is not None:
        # 亮部采样：沿 +n（亮侧）走 1.5px
        bright = bilinear_sample(dark_field, ys + ny * 1.5, xs + nx * 1.5)

    ar = np.arange(n, dtype=np.int64)
    res_a, res_b = [], []
    for d in range(1, max_gap + 1):
        qy = np.round(ys - ny * d).astype(np.int64)
        qx = np.round(xs - nx * d).astype(np.int64)
        inb = (qy >= 0) & (qy < h) & (qx >= 0) & (qx < w)
        qyi = np.clip(qy, 0, h - 1)
        qxi = np.clip(qx, 0, w - 1)
        j = idx_map[qyi, qxi].astype(np.int64)
        jsafe = np.where(j >= 0, j, 0)
        cand = inb & (j >= 0)
        if not cand.any():
            continue
        dot = nx[jsafe] * nx + ny[jsafe] * ny
        cand &= dot < dot_min
        cand &= jsafe > ar  # i < j（双向互见，仅小端建立，天然无重复对）
        if dark_field is not None:
            mid_v = bilinear_sample(dark_field, (ys + qy) * 0.5, (xs + qx) * 0.5)
            cand &= mid_v < dark_ratio * np.minimum(bright, bright[jsafe])
        if not cand.any():
            continue
        sel = np.where(cand)[0]
        res_a.append(sel)
        res_b.append(jsafe[sel])
    if not res_a:
        z = np.zeros(0, dtype=np.int64)
        return z, z
    return np.concatenate(res_a), np.concatenate(res_b)
