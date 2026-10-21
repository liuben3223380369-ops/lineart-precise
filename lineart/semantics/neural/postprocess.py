"""神经后处理链与外部语义图加载。

neural_postprocess（v1.17.1 定稿）：位置与连通性由神经图自身决定，
链路 = 对比度拉伸(下限0.2/窗0.4) → 滞回（低阈比例 0.60）→ 闭运算(3)
→ 小域清理(min_area)。neural/dexined/ensemble 三语义与外部
--semantic-map 共用本链。

外部语义图（--semantic-map）路径同样走神经后处理链：不经
verify_with_gradient 的原图梯度门控（增益门控会把纹理弱响应
放大成噪线，实测草地场景噪线 3.98%→7.38%）。
"""
import cv2
import numpy as np

from ...common import remove_small_components
from ...localization.edges import bridge_gaps, hysteresis


def _remove_small_blobs(mask, box_limit=5):
    """删除外接框 ≤ box_limit 的孤立小块（神经噪声斑块）。

    v1.17.1 移除骨架化后补的清理：滞回输出不再缩面积，实心小斑块
    （如 4x4=16px，可大于 min_area=10）无法再借骨架化缩面积后清理，
    按外接框尺寸直接判删（短于 box_limit 的双向小结构同步删除，
    与旧链骨架面积 < min_area 的清理语义对齐）。
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.ones(n, dtype=bool)
    keep[0] = False
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w <= box_limit and h <= box_limit:
            keep[i] = False
    return keep[labels]


def neural_postprocess(ink, p):
    """神经语义后处理链：位置与连通性由神经图自身决定。

    链路（v1.17.1 定稿）：对比度拉伸(下限0.2/窗0.4) → 滞回（低阈比例
    0.60，高于默认 0.45——弱响应假连接更少、覆盖更高）→ 闭运算(3)
    → 小域清理(min_area)。

    不做骨架化（v1.17.0 曾用 EDT 骨架 + 16px 断点桥接）：神经对粗线
    （≥5px）的响应是内外边缘双峰、中心谷底趋零（8px 弧线实测峰
    0.34/0.41、谷 0.001），滞回输出天然是双脊线——与 fdog 粗线同构。
    EDT 局部极大骨架在 2px 斜双线上断续（弧带命中 98%→85%）、断点
    桥接配对又删点（→80%），且 2px 双线无法再骨架化（Zhang-Suen 的
    p4·p6·p8 条件保护 2px 最小稳定结构，EDT dist<2 直通）。直接把
    双脊线交给渲染端 pair_edge_mask 配对合并取中点（fdog 粗线的
    既有设计路径，弧顶渲染为居中实心带），弧线连续性 58%→94%
    （σ0.8）/ 28%→79%（σ2.0 拍照级模糊），拐角/交叉/噪声持平。

    不走原图梯度 NMS：神经置信峰位与原图梯度峰位错开 1~4px
    （神经线宽拖尾 2~9px），二者交集仅约 11%，强联合会砍掉大部分召回。
    """
    stretch = np.clip((ink - 0.2) / 0.4, 0.0, 1.0).astype(np.float32)
    mask = hysteresis(stretch, low_ratio=0.60)
    mask = bridge_gaps(mask, closing_size=3)
    mask = remove_small_components(mask, min_area=p.min_area)
    return _remove_small_blobs(mask)


def load_semantic_map(path):
    """读取预先由神经模型生成的语义响应图（常规线稿样式：白底黑线或黑底白线）。

    输出统一为管线 ink 约定：[0,1]，1=线、0=背景。
    背景亮度由四角中值估计：白底（bg>0.5）反相，黑底恒等。
    与 Android bridge.to_semantic_map 行为一致。

    典型生成方式（任选其一，需自行安装对应环境并下载权重）：
      Anime2Sketch:  python test.py --dataroot in --output_dir out
      DexiNed:       python DexiNed.py -i in -o out
    然后把输出图路径传给 --semantic-map。
    """
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"无法读取语义图: {path}")
    gray = m.astype(np.float32) / 255.0
    h, w = gray.shape
    d = max(2, min(h, w) // 16)
    corners = [float(np.median(gray[:d, :d])), float(np.median(gray[:d, -d:])),
               float(np.median(gray[-d:, :d])), float(np.median(gray[-d:, -d:]))]
    bg = float(np.median(corners))
    sm = (1.0 - gray) if bg > 0.5 else gray
    return np.clip(sm, 0.0, 1.0).astype(np.float32)
