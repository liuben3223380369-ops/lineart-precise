"""语义层：干净灰度 → 线响应图（"哪里有线"，可插拔，不影响定位精度）。

实现：
- xdog: 多尺度软阈值 XDoG（经典）
- flow: ETF 流场 + FDoG 流线语义（Kang et al.，线条连贯）
- adaptive: Sauvola 自适应阈值（拍照手绘/扫描文档，抗光照不均）
- neural: 外部神经语义先验接口（Anime2Sketch/DexiNed 输出）
"""
from .adaptive import adaptive_semantic, sauvola_threshold
from .flow import fdog_semantic
from .neural import available, load_semantic_map
from .xdog import multiscale_xdog

__all__ = [
    "multiscale_xdog", "fdog_semantic",
    "adaptive_semantic", "sauvola_threshold",
    "available", "load_semantic_map",
]
