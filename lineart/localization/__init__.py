"""定位与渲染层：线响应图 → 亚像素线稿（"线在哪里"，精度保证）。

模块：
- gradient: 梯度算子（Sobel / Farid-Simon 5x5 最优导数）
- edges: 梯度验证 / NMS / 滞回 / 骨架 / 连通域清理 / 网格去除
- refine: 亚像素精化（抛物线 / 高斯 / Zernike）+ 中值共识滤波
- render: 抗锯齿 splat 渲染 + SSR 超采样
"""
from .edges import (
    bridge_gaps,
    extract_centerline,
    hysteresis,
    medial_skeleton,
    nonmax_suppression,
    remove_grid_bg,
    verify_with_gradient,
)
from .gradient import gradient_farid, gradient_sobel
from .refine import gaussian_fit_subpixel, zernike_subpixel_edge
from .render import refine_and_render

__all__ = [
    "gradient_farid", "gradient_sobel",
    "verify_with_gradient", "hysteresis", "nonmax_suppression",
    "bridge_gaps", "extract_centerline", "medial_skeleton",
    "remove_grid_bg",
    "gaussian_fit_subpixel", "zernike_subpixel_edge",
    "refine_and_render",
]
