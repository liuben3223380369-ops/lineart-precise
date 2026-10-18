"""底色检测与自适应反转。

CAD 截图特征：
- 黑底：黑色背景 + 彩色线条 → 需反转成白底黑线
- 白底：白色背景 + 黑色/彩色线条 → 标准流程
- 灰底：中性灰背景 → 需先分离线条与背景

策略：
1. 统计角部（4 角各 5%）像素，推断主背景色
2. 统计整体亮度分布，区分白底/黑底/灰底/混合
3. 黑底图自动反转，灰底图做背景平坦化
4. 蓝图（blueprint）：蓝底白线 → 用 V×(1-S) 提取白线后反相成标准白底黑线
"""
import cv2
import numpy as np


def _corner_stats(img, frac=0.05):
    h, w = img.shape[:2]
    dy, dx = max(1, int(h * frac)), max(1, int(w * frac))
    corners = [
        img[:dy, :dx],
        img[:dy, -dx:],
        img[-dy:, :dx],
        img[-dy:, -dx:],
    ]
    return np.concatenate([c.ravel() for c in corners])


def detect_background(img_bgr):
    """返回 (mode, need_invert)。
    mode: 'white' | 'black' | 'gray' | 'transparent' | 'mixed'
    need_invert: True 表示应反转后再走标准白底管线

    性能：所有判据（角部均值/方差、亮度直方图、蓝色占比）都是全局低频
    统计量，在 ≤320 边长的 AREA 降采样图上计算（均值/占比保持，方差因
    平均效应轻微收缩但判据余量充足），4096 图 125ms → ~10ms。
    """
    if img_bgr.shape[2] == 4:
        alpha = img_bgr[:, :, 3]
        if alpha.mean() < 200:
            return "transparent", False

    h, w = img_bgr.shape[:2]
    scale = min(1.0, 320.0 / max(h, w))
    if scale < 1.0:
        stat = cv2.resize(img_bgr, (max(1, int(round(w * scale))),
                                    max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
    else:
        stat = img_bgr

    hsv = cv2.cvtColor(stat[:, :, :3], cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # OpenCV H∈[0,180)：蓝 ≈ [95,135]（色相 190°~270°）
    blue = (h_ch >= 95) & (h_ch <= 135) & (s_ch > 70) & (v_ch > 30)
    bright_line = (v_ch > 170) & (s_ch < 90)
    if blue.mean() > 0.30 and bright_line.mean() > 0.015:
        return "blueprint", False

    gray = cv2.cvtColor(stat, cv2.COLOR_BGR2GRAY).astype(np.float32)
    corner = _corner_stats(gray)
    corner_mean = corner.mean()
    corner_std = corner.std()

    overall_mean = gray.mean()
    hist = cv2.calcHist([gray.astype(np.uint8)], [0], None, [8], [0, 255])
    hist = hist.ravel() / max(float(hist.sum()), 1e-12)
    dark_bin = hist[:2].sum()
    bright_bin = hist[-2:].sum()

    if corner_std < 12 and corner_mean > 200 and bright_bin > 0.6:
        return "white", False
    if corner_std < 12 and corner_mean < 40 and dark_bin > 0.4:
        return "black", True
    if corner_std < 20 and 60 < corner_mean < 190:
        return "gray", False
    if corner_std > 30 and overall_mean > 120 and dark_bin > 0.15:
        return "white", False
    if corner_std > 30 and overall_mean < 100 and bright_bin > 0.15:
        return "black", True

    if dark_bin > 0.45 and bright_bin > 0.45:
        return "mixed", False
    if corner_mean < 80:
        return "black", True
    if corner_mean > 170:
        return "white", False
    return "gray", False


def normalize_background(img_bgr, mode=None):
    """输入 BGR 图像，返回 (bgr_normalized, need_invert, mode)。
    输出保证是白底图（白底黑线管线入口）。
    """
    if mode is None:
        mode, _ = detect_background(img_bgr)

    if mode == "transparent":
        img = img_bgr.astype(np.float32)
        alpha = img[:, :, 3:4] / 255.0
        rgb = img[:, :, :3]
        result = (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
        return result, False, "transparent"

    if mode == "black":
        inverted = 255 - img_bgr
        return inverted, True, "black"

    if mode == "gray":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        bg = cv2.GaussianBlur(gray, (0, 0), max(gray.shape) * 0.05)
        bg = np.clip(bg, 1e-3, None)
        corrected = np.clip(gray / bg * 200.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        return result, False, "gray"

    if mode == "blueprint":
        # 蓝底白线 → 白底黑线：
        # line_score = V * (1 - S/255)：白线高（亮且不饱和），蓝底低（高饱和）
        hsv = cv2.cvtColor(img_bgr[:, :, :3], cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32) / 255.0
        s = hsv[:, :, 1].astype(np.float32) / 255.0
        line_score = v * (1.0 - s)
        ink = 1.0 - line_score  # 线→暗，底→亮
        # 背景平坦化（图纸明暗不均）
        bg = cv2.GaussianBlur(ink, (0, 0), max(ink.shape) * 0.04)
        bg = np.clip(bg, 1e-3, None)
        corrected = np.clip(ink / bg, 0.0, 1.0)
        lo = np.percentile(corrected, 2.0)
        hi = np.percentile(corrected, 98.0)
        if hi - lo > 1e-3:
            corrected = (corrected - lo) / (hi - lo)
        corrected = np.clip(corrected * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        return result, False, "blueprint"

    if mode == "white":
        return img_bgr.copy(), False, "white"

    if mode == "mixed":
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        dark = gray < 100
        result = img_bgr.copy()
        result[dark] = 255 - result[dark]
        return result, False, "mixed"

    return img_bgr.copy(), False, mode
