"""神经语义包：三语义 + 共用后处理链（v1.22.0 拆包）。

历史：v1.17.0 单文件内置 informative-drawings；v1.20.0 增 DexiNed；
v1.22.0 增 ensemble 融合并拆为四模块（单文件 655 行三模型混载，
职责膨胀）。`lineart.semantics.neural` 导入路径不变（本 __init__
re-export 全部公共接口），pipeline/CLI/GUI/测试零改动。

模块：
- informative: informative-drawings 线稿语义（semantic="neural"，
  v1.17.0；分块 + 全局归一化推理）
- dexined: DexiNed INT8 边缘语义（semantic="dexined"，v1.20.0；
  bias 反量化 + 动态尺寸两层图手术，scale 高分辨率推理 v1.21.0）
- ensemble: 三语义 noisy-OR 融合（semantic="ensemble"，v1.22.0；
  细节召回最高的推荐神经语义）
- postprocess: 神经后处理链（三语义与 --semantic-map 共用）+
  外部语义图加载

选型依据（v1.22.0 诊断 + 2000 张数据集验证，见 research_report）：
- 细节缺失主因 = informative-drawings 对弱/浅对比细节响应值塌缩
  （原始输出 <0.5 被 0.5 阈值截断；阈值 0.3→0.5 召回 0.41→0.27
  陡降，vs DexiNed 仅降 0.10）
- 召回排序（数据集实测）：ensemble > dexined > neural
  （photo +9.9/+3.5pp、lineart +7.4pp、illustration +3.2pp）
- 定位精度排序（合成圆端到端）：dexined(2x) > dexined > ensemble
  ≈ neural——追求边缘定位精度选 dexined（清晰线稿类场景配合
  dexined_scale=auto 的 2x 路由），追求细节召回选 ensemble

供应链约定：onnxruntime 与模型权重均不进核心安装，仓库不分发
权重（用户自行放置，本包不自动下载，避免供应链风险）。
"""
from .dexined import (dexined_available, dexined_semantic,
                      _DEXINED_NAME, _DEXINED_SOURCE)
from .ensemble import ensemble_available, ensemble_semantic
from .informative import (_MODEL_NAME, _MODEL_SOURCE, _find_model,
                          _infer_full, _infer_tiled, _try_ort, available,
                          _build_global_norm_session, neural_semantic)
from .postprocess import load_semantic_map, neural_postprocess

__all__ = [
    "available", "dexined_available", "ensemble_available",
    "neural_semantic", "dexined_semantic", "ensemble_semantic",
    "neural_postprocess", "load_semantic_map",
]
