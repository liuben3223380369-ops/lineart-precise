"""可选神经语义后端。

调研成果（需要下载模型权重 + PyTorch，故作为可选增强而非默认依赖）：
- Mukosame/Anime2Sketch (2135★)：对抗域适应素描提取器，动漫/插画效果佳
- carolineec/informative-drawings (CVPR'22)：无配对线稿生成
- xavysp/DexiNed (865★)：BIPED SOTA 边缘检测
- hellozhuo/pidinet (625★)：像素差分轻量边缘网络
- MengyangPu/EDTER (325★)：Transformer 边缘检测

用法：将任一模型的输出响应图（[0,1]，1=线）作为语义先验传入管线，
线条位置仍由梯度 NMS + 亚像素精化决定。本模块不自动下载权重，
避免供应链风险；用户自行放置权重文件后显式启用。
"""
import numpy as np


def _try_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def available():
    return _try_torch()


def load_semantic_map(path):
    """读取预先由神经模型生成的语义响应图（常规线稿样式：白底黑线）。

    输出统一为管线 ink 约定：[0,1]，1=线、0=背景（白底黑线反相得到）。

    典型生成方式（任选其一，需自行安装对应环境并下载权重）：
      Anime2Sketch:  python test.py --dataroot in --output_dir out
      DexiNed:       python DexiNed.py -i in -o out
    然后把输出图路径传给 --semantic-map。
    """
    import cv2
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"无法读取语义图: {path}")
    return (255.0 - m.astype(np.float32)) / 255.0
