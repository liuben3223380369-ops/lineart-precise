"""lineart-precise：高精度图片转白底黑线（自研管线）。

分层结构（v1.4.0 重组，v1.8.1 移除旧平铺 shim）：
- params.py       参数契约（LineArtParams）
- pipeline.py     主管线编排（to_lineart）
- common.py       公共工具（采样/splat/clip）
- io.py           图像读写（coerce_bgr 输入标准化）
- io/video.py     视频/动图支持（ETF 帧间一致性）
- gui/preview.py  实时预览 GUI（tkinter）
- color.py        多色线分离与颜色保留渲染
- scenes.py       场景自动检测
- __main__.py     CLI 入口（python -m lineart）
- preprocess/     预处理层（底色/光照/去噪/文字/锐化）
- semantics/      语义层（FDoG/XDoG/Sauvola/神经先验）
- localization/   定位层（梯度/NMS/精化/合并/渲染）
"""
from .params import LineArtParams
from .pipeline import to_lineart, to_lineart_batch, to_lineart_batch_simple
from .scenes import SceneInfo, detect_scene
from .semantics.adaptive import adaptive_semantic, sauvola_threshold
from .localization.gradient import gradient_farid, gradient_sobel
from .io.svg_export import raster_to_svg, to_svg_lineart, save_svg

__version__ = "1.11.5"
__all__ = [
    "LineArtParams", "to_lineart", "to_lineart_batch", "to_lineart_batch_simple",
    "gradient_farid", "gradient_sobel",
    "sauvola_threshold", "adaptive_semantic", "detect_scene", "SceneInfo",
    "raster_to_svg", "to_svg_lineart", "save_svg",
]
