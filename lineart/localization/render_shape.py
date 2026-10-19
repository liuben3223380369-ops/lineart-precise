"""外形保真渲染（shape 路线）：原图线条外形为主，管线检测为辅。

与 splat 路线（骨架 → 中心线 → 参数化宽度重绘）相对，shape 路线
保留原图线条的真实外形——宽度渐变、笔锋、末端形态直接来自原图；
管线只承担辅助角色：

- 哪里是线：滞回双阈值区域掩膜（贴合原图外形，不做 NMS/骨架化/
  边缘合并——那些操作会把外形压缩为 1px 脊线）
- 墨度浓淡与软边：梯度验证后的语义响应（亚像素斜坡，天然抗锯齿）
- 清理：最小闭运算弥合 + 小洞填充 + 小连通域剔除

宽线（> merge_gap）不受边缘配对失败退化为双线的限制；交叉点/
末端形态无重绘畸变；跳过精化/渲染两阶段，大图更快。

墨度模式（shape_ink）：
- response: 墨度 = 语义响应（边缘过渡与检测一致）
- source:   墨度 = 原图亮度反相（白点拉伸），保留原图全部笔触浓淡
- flat:     掩膜内均匀黑（最干净）
"""
import cv2
import numpy as np

from ..common import clip_f32, remove_small_components
from .edges import bridge_gaps, hysteresis


def _fill_small_holes(mask, max_area=4):
    """填充完全被线条包围的小洞（去噪可能在线内留下的白点）。

    只填不接触图像边界的背景连通域（接触边界的是真实外部背景）。
    """
    if not mask.any():
        return mask
    h, w = mask.shape
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    fill = []
    for i in range(1, n):
        left = stats[i, cv2.CC_STAT_LEFT]
        top = stats[i, cv2.CC_STAT_TOP]
        right = left + stats[i, cv2.CC_STAT_WIDTH]
        bottom = top + stats[i, cv2.CC_STAT_HEIGHT]
        touches_border = left <= 0 or top <= 0 or right >= w or bottom >= h
        if stats[i, cv2.CC_STAT_AREA] <= max_area and not touches_border:
            fill.append(i)
    if not fill:
        return mask
    out = mask.copy()
    out[np.isin(labels, fill)] = True
    return out


def build_shape_mask(strength, gray_field=None, closing=1, min_area=10):
    """外形掩膜：原图墨度 50% 深度阈值为主，管线滞回区域为辅。

    外形来源（原图为主）：
    - 墨度场 ink = 1 - gray，在滞回区域内做 Otsu 得深度阈值 t，
      外形 = ink > t——50% 半深度切分，抗锯齿边缘的面积重建无偏
      （对模糊线的过渡带取中点，与原图外形一致）
    - 管线滞回区域（strength 双阈值连通）作为范围验证：剔除非线
      暗斑（有深度但无响应），并约束 Otsu 只在真实线区取样

    保护：细化后面积 < 40%（Otsu 切割过深，如单峰细线）则回退
    滞回区域。不做 NMS / 骨架化 / 边缘合并——它们是中心线路线
    的操作（会把外形压缩为 1px 脊线）。
    """
    region = hysteresis(strength)
    if gray_field is not None and region.any():
        ink = np.subtract(1.0, gray_field.astype(np.float32, copy=False))
        vals = (ink[region] * 255.0).astype(np.uint8)
        if vals.size >= 32 and int(vals.max()) - int(vals.min()) > 8:
            # 半深度阈值：Otsu 与 0.5×p95 取大者——
            # - Otsu 对"浅裙边大峰 vs 线深小峰"的不平衡双峰会把阈值拖低
            # - 0.5×p95 是制版 50% 外形语义（AA 半像素边面积重建无偏）
            otsu_thr, _ = cv2.threshold(vals, 0, 255, cv2.THRESH_OTSU)
            half_p95 = 0.5 * float(np.percentile(vals, 95)) / 255.0
            t = max(otsu_thr / 255.0, half_p95, 0.12)
            # 外形候选 = 原图半深度区域中与管线响应区连通的连通域：
            # 原图定外形（宽线中心由原图连通性找回，DoG 对宽线中心
            # 无响应导致的滞回断裂不再是问题），管线滞回区域定取舍
            # （深度足够但不接触任何响应区的暗斑被剔除）
            cand = (ink > t).astype(np.uint8)
            n, labels = cv2.connectedComponents(cand, connectivity=8)
            if n > 1:
                touch = np.zeros(n, dtype=bool)
                touch[labels[region]] = True
                touch[0] = False
                refined = touch[labels]
                if float(refined.sum()) >= 0.4 * float(region.sum()):
                    region = refined
    if closing and int(closing) > 0:
        region = bridge_gaps(region, closing_size=int(closing))
    region = _fill_small_holes(region)
    region = remove_small_components(region, min_area=min_area)
    return region


def _geodesic_reconstruct(seed, mask, max_iters=96):
    """测地重建（by dilation）：seed 在 mask 约束内向外传播直到稳定。

    形态学测地重建 R_{k+1} = min(dilate(R_k), mask)。用于把边缘检测
    响应（DoG 对宽线中心零响应）在线条外形内传播填满——检测给出
    边界与强度，重建补全内部（"原图外形为主，检测为辅"的核心）。

    性能：只对「掩膜内响应为零」的连通域 bbox 做局部迭代（宽线
    线心），bbox 互为种子的重叠写回保持单调收敛；细线为主的图
    无零区则零成本直接返回。
    """
    m32 = mask.astype(np.float32, copy=False)
    out = np.minimum(seed.astype(np.float32, copy=False), m32)
    holes = (m32 > 0.5) & (out <= 0.01)
    if not holes.any():
        return out
    # 小碎片零区（AA 边/响应缺口，1~2px）单次 5×5 传播即可填平
    cur = cv2.dilate(out, np.ones((5, 5), np.uint8))
    np.minimum(cur, m32, out=cur)
    out = cur
    holes = (m32 > 0.5) & (out <= 0.01)
    if not holes.any():
        return out
    # 大零区（宽线线心）逐连通域 bbox 收敛重建
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(
        holes.astype(np.uint8), connectivity=8)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= 4:
            continue
        x, y, bw, bh = (int(stats[i, cv2.CC_STAT_LEFT]),
                        int(stats[i, cv2.CC_STAT_TOP]),
                        int(stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_HEIGHT]))
        y0, y1 = max(0, y - 2), min(h, y + bh + 2)
        x0, x1 = max(0, x - 2), min(w, x + bw + 2)
        ls = out[y0:y1, x0:x1]
        lm = m32[y0:y1, x0:x1]
        prev_sum = -1.0
        for _ in range(max_iters):
            cur = cv2.dilate(ls, kernel)
            np.minimum(cur, lm, out=cur)
            s = float(cur.sum())
            if s - prev_sum < 1e-3:
                break
            prev_sum = s
            ls = cur
        out[y0:y1, x0:x1] = ls
    return out


def shape_faithful_render(strength, mask, gray_field=None,
                          ink_mode="response", bg_white_pct=95.0):
    """外形保真渲染：外形掩膜 × 墨度场 → [0,1] 墨迹 canvas。

    Args:
        strength: 梯度验证后的语义响应 [0,1] float32
        mask: 外形掩膜（build_shape_mask 输出）
        gray_field: 原图亮度场 [0,1]（source 模式用，1=白）
        ink_mode: "response" | "source" | "flat"
        bg_white_pct: source 模式白点分位（掩膜外背景亮度）

    Returns:
        canvas: [0,1] float32 墨迹图（1=全墨）
    """
    if ink_mode not in ("response", "source", "flat"):
        raise ValueError(f"未知墨度模式: {ink_mode}")
    if not mask.any():
        return np.zeros(strength.shape, dtype=np.float32)

    if ink_mode == "response":
        # 边缘响应在外形内测地重建（宽线中心被两侧响应填满），
        # 边缘过渡 = 响应斜坡（1~2px 亚像素软边），模糊线的宽过渡
        # 被原样保留（这正是外形保真）
        seed = clip_f32(strength)
        if float(seed[mask].max()) < 0.05:
            ink = mask.astype(np.float32)
        else:
            ink = _geodesic_reconstruct(seed, mask)
    elif ink_mode == "source":
        if gray_field is None:
            gray_field = np.subtract(1.0, clip_f32(strength))
        bg = gray_field[~mask]
        wp = float(np.percentile(bg, bg_white_pct)) if bg.size else 1.0
        wp = max(wp, 0.05)
        ink = clip_f32((wp - gray_field) / wp) * mask
    else:  # flat
        ink = mask.astype(np.float32)
    return ink.astype(np.float32, copy=False)
