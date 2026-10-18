"""lineart-precise：高精度图片转白底黑线（自研管线）。

分层结构（第 9 轮重组）：
- params.py       参数契约（LineArtParams）
- pipeline.py     主管线编排（to_lineart）
- preprocess/     预处理层（底色/光照/去噪/文字/锐化）
- semantics/      语义层（FDoG/XDoG/Sauvola/神经先验）
- localization/   定位层（梯度/NMS/亚像素精化/渲染）
旧模块路径（subpixel/xdog/flow/... 平铺名）保留为兼容 shim。
"""
from .params import LineArtParams
from .pipeline import to_lineart
from .scenes import SceneInfo, detect_scene
from .semantics.adaptive import adaptive_semantic, sauvola_threshold
from .localization.gradient import gradient_farid, gradient_sobel

__version__ = "1.8.0"
__all__ = [
    "LineArtParams", "to_lineart", "gradient_farid", "gradient_sobel",
    "sauvola_threshold", "adaptive_semantic", "detect_scene", "SceneInfo",
]
