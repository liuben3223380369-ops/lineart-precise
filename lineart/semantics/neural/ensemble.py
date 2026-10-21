"""ensemble 三语义融合（semantic="ensemble"，v1.22.0）。

v1.22.0 细节缺失诊断（见 research_report）定位主因：neural 对弱/
浅对比细节的响应值塌缩（原始输出集中 <0.5，被 0.5 阈值截断）；
dexined 响应健康但极浅对比细线召回 @0.5 仅 0.86~0.89。三者
noisy-OR 概率融合让任一分量的响应都可透传、弱响应叠加也能过阈：

    ink = 1 - (1-dexined)(1-neural)(1-梯度)

实测（vs dexined 单语义端到端）：
- 弱对比山脊召回 0.919→0.992（+7.3pp）
- 极浅对比 1px 细线召回 0.858→0.975（+11.7pp）
- 纹理噪线可控：端到端墨线总量仅 +17~32%（滞回+连通域清理
  压制散点；纹理背景原始误报 1.57% → 端到端无噪线放大）

2000 张公开数据集验证（COCO 照片 / anime 插画 / 动漫线稿 /
素描配对各 500）：全部三类真实数据召回最优（photo +9.9pp /
lineart +7.4pp / illustration +3.2pp）——追求细节召回的默认建议。
"""
import cv2
import numpy as np

from .dexined import (_DEXINED_NAME, dexined_available, dexined_semantic)
from .informative import _MODEL_NAME, available, neural_semantic


def ensemble_available():
    """ensemble 融合语义是否可用（双权重齐备）。"""
    return available() and dexined_available()


def _gradient_semantic(bgr):
    """Sobel 梯度幅值语义（归一化 [0,1]），纯 OpenCV 无额外依赖。

    ensemble 融合的高频细节分量：神经模型漏检的浅对比细线在
    梯度响应上仍然完整（细线召回 0.60 vs neural 0.26），
    由 noisy-OR 的概率叠加机制送过阈值。

    v1.22.0 移除 1024 下采样：实测全分辨率 Sobel 反而更快
    （5ms vs 18ms，下采样+回放本身有开销）且无精度差——
    原下采样是无效的速度优化。
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, 3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, 3)
    mag = np.sqrt(gx * gx + gy * gy)
    return np.clip(mag / (mag.max() + 1e-6), 0, 1).astype(np.float32)


def ensemble_semantic(img_bgr, model_path=None, scale=1):
    """ensemble 三语义 noisy-OR 融合 → ink 语义图（[0,1]，1=线）。

    分量：dexined_semantic（精度基准）+ neural_semantic（结构召回）
    + Sobel 梯度（高频细节）。分量间尺寸对齐后按
    1 - ∏(1-mᵢ) 融合（见模块 docstring 实测数据）。

    scale 语义与 dexined_semantic 一致（1/2，传递给 dexined 分量）。
    需双权重（lineart.onnx + dexined.onnx）齐备；缺失时报错指路。
    Session 复用各自缓存，重复调用零加载开销。
    """
    if not ensemble_available():
        raise FileNotFoundError(
            "ensemble 融合语义需双权重齐备：informative-drawings "
            f"（{_MODEL_NAME}，semantic=\"neural\" 用）与 DexiNed "
            f"（{_DEXINED_NAME}，semantic=\"dexined\" 用）。"
            "放置说明见 README「神经能力」一节。")
    bgr = np.asarray(img_bgr)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr * 255.0 if bgr.max() <= 1.0 else bgr,
                      0, 255).astype(np.uint8)
    h, w = bgr.shape[:2]
    maps = []
    for m in (dexined_semantic(bgr, model_path, scale=scale),
              neural_semantic(bgr, model_path),
              _gradient_semantic(bgr)):
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h))
        maps.append(m)
    keep = np.ones((h, w), np.float32)
    for m in maps:
        keep *= (1.0 - m)
    return np.clip(1.0 - keep, 0.0, 1.0).astype(np.float32)
