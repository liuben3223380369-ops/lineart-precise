"""A2 线条连通性后处理：桥接断裂线条端点。

算法：
1. 找到所有端点（邻域只有一个前景像素的点）
2. 空间网格（边长 max_gap）分桶，只对邻桶内的端点对计算距离，
   生成 dist ∈ (1, max_gap] 的候选对（i < j）
3. 按 (i 升序, dist 升序) 贪心扫描：每个端点最多参与一次连接，
   与原 O(N²) 逐端点暴力搜索语义完全等价
4. 用 Bresenham 直线连接
5. 连接后重新做骨架化

用于修复滞回阈值未完全连接的弱边缘、细弱线断点、纹理噪声导致的
局部断裂。距离约束桥接（方向一致性信息经实测收益不抵 ETF 开销，
v1.9.2 起不再计算）。

v1.9.2 性能修复：端点匹配由 Python O(N²) 标量循环（68k 端点 65s）
改为空间网格 + 向量化距离 + 排序贪心（<0.1s）；同时修复 Bresenham
斜率未累加导致桥接线退化为水平/垂直线的正确性缺陷。
"""
from collections import defaultdict

import cv2
import numpy as np

from ..common import remove_small_components
from .edges import bridge_gaps, medial_skeleton


def _find_endpoints(mask):
    """找到所有端点：邻域（8 连通）中只有一个前景像素的点。"""
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int64)

    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(m, cv2.CV_32F, kernel)
    endpoints = (neighbor_count == 2) & m
    ys, xs = np.nonzero(endpoints)
    return np.column_stack([ys, xs])


def _bresenham_line(y0, x0, y1, x1):
    """Bresenham 直线算法（返回 [y, x] 点序列）。"""
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            y0, x0, y1, x1 = y1, x1, y0, x0
        if x1 == x0:
            return np.array([[y0, x0]], dtype=np.int64)
        slope = (y1 - y0) / (x1 - x0)
        points = [[int(round(y0 + slope * (x - x0))), x]
                  for x in range(x0, x1 + 1)]
    else:
        if y0 > y1:
            y0, x0, y1, x1 = y1, x1, y0, x0
        if y1 == y0:
            return np.array([[y0, x0]], dtype=np.int64)
        slope = (x1 - x0) / (y1 - y0)
        points = [[y, int(round(x0 + slope * (y - y0)))]
                  for y in range(y0, y1 + 1)]

    return np.array(points, dtype=np.int64)


def _candidate_pairs(ys, xs, max_gap):
    """空间网格生成候选端点对，返回 (i, j, dist) 数组（i < j，1 < dist <= max_gap）。"""
    n = len(ys)
    if n < 2:
        return np.empty((0, 3), dtype=np.float64)

    g = max(1, int(max_gap))
    buckets = defaultdict(list)
    for idx in range(n):
        buckets[(ys[idx] // g, xs[idx] // g)].append(idx)

    left_i = []
    left_j = []
    # 半邻域遍历（右 / 左下 / 下 / 右下）保证每对桶只访问一次
    offsets = ((0, 1), (1, -1), (1, 0), (1, 1))
    for (ay, ax), members in buckets.items():
        for dy, dx in offsets:
            other = buckets.get((ay + dy, ax + dx))
            if other:
                for i in members:
                    for j in other:
                        left_i.append(i)
                        left_j.append(j)
        if len(members) > 1:
            m = np.asarray(members)
            ii, jj = np.triu_indices(len(m), 1)
            left_i.extend(m[ii].tolist())
            left_j.extend(m[jj].tolist())

    if not left_i:
        return np.empty((0, 3), dtype=np.float64)

    pi = np.asarray(left_i, dtype=np.int64)
    pj = np.asarray(left_j, dtype=np.int64)
    keep = pi < pj
    pi, pj = pi[keep], pj[keep]
    if len(pi) == 0:
        return np.empty((0, 3), dtype=np.float64)

    fy = ys.astype(np.float64)
    fx = xs.astype(np.float64)
    dist = np.hypot(fy[pj] - fy[pi], fx[pj] - fx[pi])
    ok = (dist > 1.0) & (dist <= max_gap)
    pairs = np.column_stack([pi[ok], pj[ok], dist[ok]])
    return pairs


def bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8, min_area=10):
    """桥接断裂线条端点（空间网格加速，贪心匹配与暴力版语义等价）。"""
    if max_gap <= 1 or not mask.any():
        return mask

    endpoints = _find_endpoints(mask)
    if len(endpoints) < 2:
        return mask

    ys = endpoints[:, 0].astype(np.int64)
    xs = endpoints[:, 1].astype(np.int64)
    pairs = _candidate_pairs(ys, xs, max_gap)

    result = mask.astype(np.uint8)
    if len(pairs):
        # (i 升序, dist 升序)：对每个 i 依距离序选首个未连接 j，等价于原贪心
        order = np.lexsort((pairs[:, 2], pairs[:, 0]))
        pairs = pairs[order]
        pi = pairs[:, 0].astype(np.int64)
        pj = pairs[:, 1].astype(np.int64)
        used = np.zeros(len(endpoints), dtype=bool)
        for k in range(len(pairs)):
            i, j = pi[k], pj[k]
            if used[i] or used[j]:
                continue
            used[i] = used[j] = True
            pts = _bresenham_line(int(ys[i]), int(xs[i]), int(ys[j]), int(xs[j]))
            result[pts[:, 0], pts[:, 1]] = 1

    # 形态学闭运算弥合小间隙（在骨架化之前，避免骨架化后闭运算破坏中心线）
    result = bridge_gaps(result, closing_size=1)

    # 重新骨架化
    result = medial_skeleton(result.astype(bool))

    # 清理小连通域
    result = remove_small_components(result, min_area=min_area)

    return result
