"""参数定义：LineArtParams（管线全部可调参数的唯一来源）。

从 pipeline.py 拆出（第 9 轮结构重组）：pipeline 只做编排，
参数契约独立成模块，便于 CLI / API / 测试共享。
"""
from dataclasses import dataclass


@dataclass
class LineArtParams:
    mode: str = "antialiased"
    semantic: str = "fdog"
    output: str = "grayscale"
    detail: float = 0.6
    line_sensitivity: float = 0.5
    thickness: float = 1.0
    denoise: float = 0.4
    illumination: float = 0.8
    coherence: float = 0.6
    smooth_radius: int = 3
    min_area: int = 10
    max_side: int = 4096
    gradient_gamma: float = 0.7
    n_colors: int = 8
    remove_grid: bool = False
    grid_period_min: int = 2
    remove_text: bool = False
    extract_centerline: bool = False
    auto_invert: bool = True
    gradient_method: str = "farid"
    subpixel_method: str = "gaussian"
    sharpen: float = 0.0  # 相干冲击锐化强度（0=off，1=全量；模糊输入推荐 0.5~0.8）
    render_ss: int = 2   # 渲染超采样倍率（2=2x 画布 splat 后 INTER_AREA 下采样）
    consensus: bool = True  # 亚像素偏移邻域一致性滤波（剔除拟合离群）
    merge_edges: bool = True  # 粗线双边缘合并为线中心（结构保真）
    merge_gap: int = 6  # 边缘对合并的最大搜索距离（像素，覆盖线宽 ~7px）
    scene: str = "auto"  # auto | off | pixel | blueprint | sketch | lineart | photo
