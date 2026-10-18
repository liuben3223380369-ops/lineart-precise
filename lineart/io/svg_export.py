# -*- coding: utf-8 -*-
"""
文件名: svg_export.py
描述: SVG 矢量输出：二值掩码 → 连通域 → 轮廓 → 道格拉斯-普克简化 → SVG path
"""
import cv2
import numpy as np


def _douglas_peucker(points, epsilon):
    """道格拉斯-普克算法：折线简化，保留关键拐点。

    参数:
        points: (N, 2) 数组，每行为 (x, y)
        epsilon: 简化容差（像素），越大越简化
    返回:
        简化后的 (M, 2) 数组
    """
    if len(points) <= 2:
        return points

    # 找到距离首尾连线最远的点
    start, end = points[0], points[-1]
    line_vec = end - start
    line_len = np.hypot(line_vec[0], line_vec[1])

    if line_len < 1e-9:
        # 首尾重合，返回闭合路径
        return points[:1]

    # 计算每个点到首尾连线的垂直距离
    # 叉积 |(P - start) × line_vec| / |line_vec|
    rel = points - start
    cross = np.abs(rel[:, 0] * line_vec[1] - rel[:, 1] * line_vec[0])
    dists = cross / line_len

    # 忽略首尾点
    dists[0] = dists[-1] = 0
    max_idx = int(np.argmax(dists))
    max_dist = dists[max_idx]

    if max_dist > epsilon:
        # 递归简化两段
        left = _douglas_peucker(points[:max_idx + 1], epsilon)
        right = _douglas_peucker(points[max_idx:], epsilon)
        # 合并（避免重复中间点）
        return np.vstack([left[:-1], right])
    else:
        return np.array([start, end])


def _contour_to_svg_path(contour, simplify_tol=1.0):
    """将单个轮廓转换为 SVG path 字符串（绝对坐标）。

    参数:
        contour: (N, 1, 2) OpenCV 轮廓格式
        simplify_tol: 道格拉斯-普克简化容差
    返回:
        SVG path 字符串，如 "M 10 20 L 30 40 L 50 60 Z"
    """
    # 转换为 (N, 2) 并取浮点坐标
    pts = contour.reshape(-1, 2).astype(np.float64)

    if len(pts) < 3:
        # 少于 3 点无法形成闭合路径，跳过
        return None

    # 闭合轮廓：确保首尾相同
    if not np.array_equal(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    # 道格拉斯-普克简化
    simplified = _douglas_peucker(pts, simplify_tol)

    if len(simplified) < 3:
        return None

    # 构建 SVG path（绝对坐标）
    # M = moveto, L = lineto, Z = closepath
    parts = [f"M {simplified[0][0]:.2f} {simplified[0][1]:.2f}"]
    for pt in simplified[1:]:
        parts.append(f"L {pt[0]:.2f} {pt[1]:.2f}")
    parts.append("Z")

    return " ".join(parts)


def raster_to_svg(binary_mask, simplify_tol=1.0):
    """将二值掩码转换为 SVG 字符串。

    参数:
        binary_mask: 二值掩码，白底黑线（255=白背景，0=黑线条）
        simplify_tol: 道格拉斯-普克简化容差（像素），默认 1.0
    返回:
        SVG 字符串
    """
    if binary_mask is None or binary_mask.size == 0:
        return _empty_svg(0, 0)

    h, w = binary_mask.shape[:2]

    # 确保是二值图：线条为黑色(0)，背景为白色(255)
    # 输入约定：白底黑线，所以用线条区域（黑色部分）
    # 提取线条区域（黑色像素）
    if binary_mask.ndim == 3:
        gray = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
    else:
        gray = binary_mask

    # 线条是黑色(0)，需要反转来找轮廓（findContours 找白色区域）
    # 所以反转：线条变白色，背景变黑色
    inverted = cv2.bitwise_not(gray)

    # 查找外部轮廓（RETR_EXTERNAL 只取最外层，避免嵌套）
    # RETR_LIST 或 RETR_CCOMP 也可，这里用 RETR_TREE 获取层次
    contours, hierarchy = cv2.findContours(
        inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return _empty_svg(w, h)

    # 转换每个轮廓为 SVG path
    paths = []
    for contour in contours:
        # 过滤太小的轮廓（面积阈值）
        area = cv2.contourArea(contour)
        if area < 1.0:
            continue

        path_str = _contour_to_svg_path(contour, simplify_tol)
        if path_str:
            paths.append(path_str)

    if not paths:
        return _empty_svg(w, h)

    # 构建 SVG
    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'  <rect width="100%" height="100%" fill="white"/>',
        f'  <g stroke="black" fill="none" stroke-width="1">',
    ]
    for path in paths:
        svg_lines.append(f'    <path d="{path}"/>')
    svg_lines.extend([
        '  </g>',
        '</svg>',
    ])

    return '\n'.join(svg_lines)


def _empty_svg(width, height):
    """生成空 SVG（白色背景）。"""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        f'  <rect width="100%" height="100%" fill="white"/>\n'
        f'</svg>'
    )


def to_svg_lineart(img_bgr, params=None, simplify_tol=1.0):
    """将 BGR 图像转换为 SVG 线稿。

    参数:
        img_bgr: BGR 图像
        params: LineArtParams 参数，为 None 时使用默认值
        simplify_tol: 道格拉斯-普克简化容差（像素）
    返回:
        SVG 字符串
    """
    from ..pipeline import to_lineart
    from ..params import LineArtParams

    p = params or LineArtParams()
    # 确保输出为灰度模式（白底黑线）
    p = LineArtParams(
        **{**p.__dict__, "output": "grayscale", "mode": "antialiased"}
    )

    # 获取线稿结果
    result = to_lineart(img_bgr, p)

    # 转换为 SVG
    svg = raster_to_svg(result, simplify_tol=simplify_tol)
    return svg


def save_svg(svg_string, filepath):
    """保存 SVG 字符串到文件。

    参数:
        svg_string: SVG 字符串
        filepath: 输出文件路径
    """
    import os

    # 确保扩展名为 .svg
    if not str(filepath).lower().endswith('.svg'):
        filepath = str(filepath) + '.svg'

    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(svg_string)
