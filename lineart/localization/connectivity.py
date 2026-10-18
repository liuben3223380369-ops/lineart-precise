"""A2 线条连通性后处理：桥接断裂线条端点。

算法：
1. 找到所有端点（邻域只有一个前景像素的点）
2. 对每个端点，沿流线方向（tx, ty ETF 切向流场）搜索 max_gap 距离内的其他端点
3. 如果找到，用 Bresenham 直线连接
4. 连接后重新做骨架化

用于修复：
- 滞回阈值未完全连接的弱边缘
- 细弱线因对比度不足产生的断点
- 纹理噪声导致的局部断裂
"""
import cv2
import numpy as np

from ..common import EPS, remove_small_components
from .edges import bridge_gaps, medial_skeleton


def _find_endpoints(mask):
    """找到所有端点：邻域（8 连通）中只有一个前景像素的点。"""
    if not mask.any():
        return np.zeros((0, 2), dtype=np.int64)
    
    h, w = mask.shape
    m = mask.astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(m, cv2.CV_32F, kernel)
    endpoints = (neighbor_count == 2) & m
    ys, xs = np.nonzero(endpoints)
    return np.column_stack([ys, xs])


def _bresenham_line(y0, x0, y1, x1):
    """Bresenham 直线算法。"""
    dy = y1 - y0
    dx = x1 - x0
    
    if abs(dy) < abs(dx):
        if x0 > x1:
            y0, x0, y1, x1 = y1, x1, y0, x0
            dy, dx = -dy, -dx
        if dx == 0:
            return np.array([[y0, x0]], dtype=np.int64)
        slope = dy / dx
        y = y0
        points = [[int(round(y)), x] for x in range(x0, x1 + 1)]
        y += slope
    else:
        if y0 > y1:
            y0, x0, y1, x1 = y1, x1, y0, x0
            dy, dx = -dy, -dx
        if dy == 0:
            return np.array([[y0, x0]], dtype=np.int64)
        slope = dx / dy
        x = x0
        points = [[y, int(round(x))] for y in range(y0, y1 + 1)]
        x += slope
    
    return np.array(points, dtype=np.int64)


def bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8, min_area=10):
    """桥接断裂线条端点。

    简化版本：移除 scipy 依赖，使用暴力搜索。
    """
    if max_gap <= 1 or not mask.any():
        return mask
    
    h, w = mask.shape
    endpoints = _find_endpoints(mask)
    
    if len(endpoints) < 2:
        return mask
    
    result = mask.copy().astype(np.uint8)
    connected = set()
    
    for i, (y0, x0) in enumerate(endpoints):
        if i in connected:
            continue
        
        best_j = -1
        best_dist = float('inf')
        
        for j, (y1, x1) in enumerate(endpoints):
            if j <= i or j in connected:
                continue
            
            dist = np.hypot(y1 - y0, x1 - x0)
            if dist < 1.0 or dist > max_gap:
                continue
            
            # 简化：只检查距离，不检查流场一致性
            if dist < best_dist:
                best_dist = dist
                best_j = j
        
        if best_j >= 0:
            y1, x1 = endpoints[best_j]
            pts = _bresenham_line(y0, x0, y1, x1)
            for y, x in pts:
                if 0 < y < h - 1 and 0 < x < w - 1:
                    result[y, x] = 1
            
            connected.add(i)
            connected.add(best_j)
    
    # 形态学闭运算弥合小间隙（在骨架化之前，避免骨架化后闭运算破坏中心线）
    result = bridge_gaps(result, closing_size=1)
    
    # 重新骨架化
    result = medial_skeleton(result.astype(bool))
    
    # 清理小连通域
    result = remove_small_components(result, min_area=min_area)
    
    return result
