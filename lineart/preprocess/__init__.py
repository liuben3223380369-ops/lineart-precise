"""预处理层：输入图像 → 干净灰度（光照/去噪/底色/文字/锐化）。

管线阶段顺序（to_lineart 内的调用序）：
    底色检测反转(background) → 光照归一化(illumination) →
    保边去噪(denoise) → 文字去除(text, 可选) → 冲击锐化(shock, 可选)
"""
from .background import normalize_background
from .denoise import edge_preserving_denoise, guided_filter
from .illumination import estimate_background, normalize_illumination
from .shock import coherence_shock, shock_sharpen
from .text import detect_text_mser, inpaint_text_regions

__all__ = [
    "normalize_background", "guided_filter", "edge_preserving_denoise",
    "estimate_background", "normalize_illumination",
    "coherence_shock", "shock_sharpen",
    "detect_text_mser", "inpaint_text_regions",
]
