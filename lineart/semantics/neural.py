"""可选神经语义后端。

调研清单（v1.16.0 调研成果，均需下载模型权重）：
- carolineec/informative-drawings (CVPR'22)：无配对线稿生成 ← v1.17.0 内置
- Mukosame/Anime2Sketch (2135★)：对抗域适应素描提取器，动漫/插画效果佳
- xavysp/DexiNed (865★)：BIPED SOTA 边缘检测
- hellozhuo/pidinet (625★)：像素差分轻量边缘网络
- MengyangPu/EDTER (325★)：Transformer 边缘检测

v1.17.0 起内置 informative-drawings lineart 风格 ONNX 推理：
神经后处理链（对比度拉伸 + 滞回直通 + 断点桥接）在弱结构边缘上的
召回率远超内建梯度语义（合成场景实测：弱对比山脊 4.6%→76%，
纹理背景中轮廓 13%→90%，视觉评分 4→6.5 / 5.5→8.5）。

仍作为可选增强而非默认依赖：onnxruntime 与 17MB 权重不进核心安装，
用户自行放置权重后显式启用（semantic="neural"）。本模块不自动下载
权重，避免供应链风险。

外部语义图（--semantic-map）路径同样走神经后处理链：位置与连通性
由神经图自身决定，不经 verify_with_gradient 的原图梯度门控（增益
门控会把纹理弱响应放大成噪线，实测草地场景噪线 3.98%→7.38%）。
"""
import os
from pathlib import Path

import cv2
import numpy as np

_MODEL_NAME = "lineart.onnx"
_MODEL_SOURCE = ("https://hf-mirror.com/rocca/"
                 "informative-drawings-line-art-onnx (resolve/main/lineart.onnx)")

_SESSION = {"path": None, "sess": None}


def _candidate_paths(model_path=None):
    if model_path:
        return [Path(model_path)]
    paths = []
    env = os.environ.get("LINEART_NEURAL_MODEL")
    if env:
        paths.append(Path(env))
    paths.append(Path.home() / ".lineart" / _MODEL_NAME)
    paths.append(Path(__file__).resolve().parent.parent.parent
                 / "models" / _MODEL_NAME)
    return paths


def _find_model(model_path=None):
    for p in _candidate_paths(model_path):
        if p.is_file():
            return str(p)
    return None


def _try_ort():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def available():
    """神经语义是否可用（onnxruntime 已装 + 权重可寻）。"""
    return _try_ort() and _find_model() is not None


def neural_semantic(img_bgr, model_path=None):
    """informative-drawings lineart ONNX 推理 → ink 语义图（[0,1]，1=线）。

    输入 BGR/灰度（uint8 或 float [0,1]），内部转 NCHW RGB [0,1]；
    模型输出 1=背景、0=线，反相统一为管线 ink 约定（1=线）。
    Session 按权重路径缓存（重复调用零开销）。
    """
    import onnxruntime as ort
    path = _find_model(model_path)
    if path is None:
        raise FileNotFoundError(
            "未找到神经语义模型权重（semantic=\"neural\"）。请将 "
            f"informative-drawings lineart ONNX（约 17MB，文件名 {_MODEL_NAME}）"
            "放置于以下任一位置：① 环境变量 LINEART_NEURAL_MODEL 指定的路径；"
            "② ~/.lineart/lineart.onnx；③ 仓库根目录 models/lineart.onnx；"
            "或通过 params.neural_model / CLI --neural-model 显式指定。"
            f"权重来源：{_MODEL_SOURCE}")
    if _SESSION["path"] != path:
        _SESSION["sess"] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        _SESSION["path"] = path
    bgr = np.asarray(img_bgr)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr * 255.0 if bgr.max() <= 1.0 else bgr,
                      0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    y = _SESSION["sess"].run(["output"], {"input": x})[0][0, 0]
    return np.clip(1.0 - y, 0.0, 1.0).astype(np.float32)


def neural_postprocess(ink, p):
    """神经语义后处理链：位置与连通性由神经图自身决定。

    链路（合成场景实验定稿）：对比度拉伸(下限0.2/窗0.4) → 滞回
    （低阈比例 0.60，高于默认 0.45——弱响应假连接更少、覆盖更高）
    → 闭运算(3) → 骨架化 → 断点桥接(16px，神经拖尾处断线修复实测
    bg16 视觉断线显著少于 bg8) → 小域清理(min_area)。

    不走原图梯度 NMS：神经置信峰位与原图梯度峰位错开 1~4px
    （神经线宽拖尾 2~9px），二者交集仅约 11%，强联合会砍掉大部分召回。
    """
    from ..common import remove_small_components
    from ..localization.connectivity import bridge_broken_lines
    from ..localization.edges import bridge_gaps, hysteresis, medial_skeleton

    stretch = np.clip((ink - 0.2) / 0.4, 0.0, 1.0).astype(np.float32)
    mask = hysteresis(stretch, low_ratio=0.60)
    mask = bridge_gaps(mask, closing_size=3)
    mask = medial_skeleton(mask)
    mask = bridge_broken_lines(mask, None, None, None, None, max_gap=16)
    mask = remove_small_components(mask, min_area=p.min_area)
    return mask


def load_semantic_map(path):
    """读取预先由神经模型生成的语义响应图（常规线稿样式：白底黑线）。

    输出统一为管线 ink 约定：[0,1]，1=线、0=背景（白底黑线反相得到）。

    典型生成方式（任选其一，需自行安装对应环境并下载权重）：
      Anime2Sketch:  python test.py --dataroot in --output_dir out
      DexiNed:       python DexiNed.py -i in -o out
    然后把输出图路径传给 --semantic-map。
    """
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"无法读取语义图: {path}")
    return (255.0 - m.astype(np.float32)) / 255.0
