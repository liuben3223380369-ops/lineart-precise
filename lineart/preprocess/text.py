"""文字区域检测（CAD 标注文字）。

CAD 截图中的文字特征：
- 水平或垂直方向
- 单色、高对比度
- 字体大小相对均匀
- 与线条无重叠（或轻微重叠）

策略：
1. MSER 默认：纯 OpenCV，无需模型，检测稳定文字区域
2. 可选 EAST 深度学习检测器（需下载 frozen_east_text_detection.pb）
3. 输出文字掩码，供管线选择保留或去除
"""
import cv2
import numpy as np


def detect_text_mser(gray, min_area=10, max_area_ratio=0.01,
                     min_aspect=0.15, max_aspect=3.0):
    """文字区域检测：二值化 + 形态学膨胀 + 连通域筛选。返回文字掩码（bool）。

    CAD 文字特征：白色/亮色、小面积、长宽比适中。
    抗锯齿导致轮廓碎片化，改用形态学膨胀合并相邻碎片。
    """
    h, w = gray.shape
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5))
    dilated = cv2.dilate(bright, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        dilated, connectivity=8)

    mask = np.zeros((h, w), dtype=bool)
    max_area = max(100, int(h * w * max_area_ratio))
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        rw = stats[i, cv2.CC_STAT_WIDTH]
        rh = stats[i, cv2.CC_STAT_HEIGHT]
        if rw < 4 or rh < 4:
            continue
        aspect = rw / max(rh, 1)
        if aspect < min_aspect or aspect > max_aspect:
            continue
        mask[labels == i] = True

    return mask


def detect_text_east(gray, model_path=None, conf_thresh=0.5, nms_thresh=0.4):
    """EAST 深度学习文字检测（需模型文件）。返回文字掩码。"""
    h, w = gray.shape
    if gray.ndim == 2:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        rgb = gray

    net = cv2.dnn.readNet(model_path)
    blob = cv2.dnn.blobFromImage(rgb, 1.0, (w, h),
                                  (123.68, 116.78, 103.94),
                                  swapRB=True, crop=False)
    net.setInput(blob)
    layers = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]
    scores, geometry = net.forward(layers)

    _, nH = scores.shape[2:4]
    rects, confidences = [], []
    for y in range(nH):
        scores_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]
        for x in range(w // 4):
            if scores_data[x] < conf_thresh:
                continue
            offset_x, offset_y = x * 4.0, y * 4.0
            angle = angles[x]
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            h_val = x_data0[x] + x_data2[x]
            w_val = x_data1[x] + x_data3[x]
            end_x = int(offset_x + cos_a * x_data1[x] + sin_a * x_data2[x])
            end_y = int(offset_y - sin_a * x_data1[x] + cos_a * x_data2[x])
            start_x = int(end_x - w_val)
            start_y = int(end_y - h_val)
            rects.append((start_x, start_y, end_x, end_y))
            confidences.append(scores_data[x])

    if not rects:
        return np.zeros((h, w), dtype=bool)
    boxes = cv2.dnn.NMSBoxesRotated(rects, confidences, conf_thresh, nms_thresh)
    mask = np.zeros((h, w), dtype=bool)
    if len(boxes) > 0:
        for i in boxes.flatten():
            rx, ry, rw, rh = cv2.boundingRect(cv2.boxPoints(rects[i]))
            rx = max(0, rx - 2)
            ry = max(0, ry - 2)
            rw = min(w - rx, rw + 4)
            rh = min(h - ry, rh + 4)
            mask[ry:ry+rh, rx:rx+rw] = True
    return mask


def inpaint_text_regions(img, text_mask, radius=3):
    """用 Navier-Stokes inpainting 去除文字区域，保留周围线条。"""
    mask_u8 = (text_mask * 255).astype(np.uint8)
    return cv2.inpaint(img, mask_u8, radius, cv2.INPAINT_NS)
