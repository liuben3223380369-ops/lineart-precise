"""参数定义：LineArtParams（管线全部可调参数的唯一来源）。

从 pipeline.py 拆出（第 9 轮结构重组）：pipeline 只做编排，
参数契约独立成模块，便于 CLI / API / 测试共享。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class LineArtParams:
    mode: str = "antialiased"
    semantic: str = "fdog"  # fdog | xdog | adaptive | neural（ONNX 神经先验，需自备权重）| dexined（OpenCV INT8 边缘检测，精度更高）
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
    merge_gap: int = 20  # 边缘对合并最大搜索距离（像素，覆盖线宽 ~21px；v1.15.0：6→20，旧值下 8px+ 粗线双边缘配对失败退化为双线）
    bridge_gap: int = 8  # 断点桥接的最大距离（像素，沿 ETF 流场方向搜索）
    scene: str = "auto"  # auto | off | pixel | blueprint | sketch | lineart | photo
    adaptive_width: bool = True  # 自适应线宽：根据线类型（硬边/软线）调整 FWHM 阈值
    uniform_width: bool = False  # 统一线宽（v1.16.1）：合并粗线不按原图 FWHM 线宽展开，全部按 thickness 渲染（CAD 无级差线稿）
    render_mode: str = "splat"  # 渲染路线：splat 中心线重绘 | shape 外形保真（原图外形为主）
    shape_ink: str = "response"  # shape 路线墨度：response 语义响应 | source 原图浓淡 | flat 均匀黑
    shape_closing: int = 1  # shape 路线外形清理的闭运算尺寸（1=最小，保外形；0=关闭）
    neural_model: Optional[str] = None  # semantic="neural"/"dexined" 时的 ONNX 权重路径（None=自动搜索：环境变量 LINEART_NEURAL_MODEL/LINEART_DEXINED_MODEL / ~/.lineart/ 对应 onnx / 仓库 models/）
