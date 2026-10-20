"""多色线分离与颜色保留渲染。

CAD 截图特征：
- 不同图层/线型用不同颜色（红/黄/绿/蓝/青/品红等）
- 线条细（0.5-2px），颜色饱和度高
- 背景通常为黑/灰/白

策略：
1. 颜色空间聚类：在 Lab 空间聚类，分离不同颜色的线条
2. 主色提取：按色相分组，提取前 N 个主色
3. 颜色保留渲染：每个颜色通道独立做线稿提取，然后按颜色合成
4. 输出模式：
   - "color"：保留原始颜色的白底彩色线稿
   - "grayscale"：白底黑线（标准）
   - "layered"：分层输出，每层一个颜色
"""
import cv2
import numpy as np

from .common import remove_small_components


def extract_color_layers(img_bgr, n_colors=8, min_sat=20, min_val=30):
    """在 Lab + HSV 空间分离彩色线条。

    返回 (layers, colors)，其中：
    - layers: list of (mask, color_bgr)，mask 是该颜色的二值掩码
    - colors: list of (B,G,R) 主色
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    hue = hsv[:, :, 0].astype(np.float32)

    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    is_color = (sat > min_sat) & (val > min_val)
    is_dark = val < 100

    pixels = np.stack([a_ch, b_ch], axis=-1).reshape(-1, 2)
    color_mask = is_color.reshape(-1)
    dark_mask = is_dark.reshape(-1)

    valid = color_mask & dark_mask
    valid_pixels = pixels[valid]

    if len(valid_pixels) < 100:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return [(gray < 100, (0, 0, 0))], [(0, 0, 0)]

    n_clusters = min(n_colors, max(1, len(valid_pixels) // 50))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        valid_pixels.astype(np.float32), n_clusters, None, criteria, 3,
        cv2.KMEANS_PP_CENTERS
    )

    layers = []
    colors = []
    for i, center in enumerate(centers):
        dist = np.sqrt(np.sum((pixels - center) ** 2, axis=-1))
        prob = np.exp(-dist / 2000.0).reshape(img_bgr.shape[:2])

        mask = (prob > 0.3) & is_color & is_dark
        mask = mask.astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        if mask.sum() < 200:
            continue

        yx = np.nonzero(mask > 0)
        if len(yx[0]) == 0:
            continue
        bgr_vals = img_bgr[yx].mean(axis=0)
        bgr_list = bgr_vals.tolist()
        colors.append(tuple(int(round(float(x))) for x in bgr_list))
        layers.append((mask > 0, tuple(int(round(float(x))) for x in bgr_list)))

    if not layers:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return [(gray < 100, (0, 0, 0))], [(0, 0, 0)]

    return layers, colors


def render_color_lineart(layers, canvas_size, thickness=1.0):
    """按颜色分层渲染：每层独立做线稿提取后按颜色合成到白底。

    返回 BGR 白底彩色线稿。
    """
    h, w = canvas_size
    canvas = np.full((h, w, 3), 255, dtype=np.uint8)

    for mask, color_bgr in layers:
        if not mask.any():
            continue
        m = (mask > 0).astype(np.uint8) * 255
        if thickness > 1.0:
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (max(1, int(round(thickness))), max(1, int(round(thickness)))))
            m = cv2.dilate(m, k)

        color_arr = np.array(color_bgr, dtype=np.uint8)
        for c in range(3):
            canvas[:, :, c][m > 0] = color_arr[c]

    return canvas


def to_color_lineart(img_bgr, layers=None, n_colors=8, thickness=1.0):
    """输出白底彩色线稿。"""
    if layers is None:
        layers, _ = extract_color_layers(img_bgr, n_colors=n_colors)
    return render_color_lineart(layers, img_bgr.shape[:2], thickness=thickness)
