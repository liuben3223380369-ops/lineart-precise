"""光照归一化：消除阴影与亮度渐变，保证不同光照下线条判定一致。"""
import cv2
import numpy as np


def estimate_background(gray, downscale=0.125, sigma_ratio=0.035):
    """背景低频场估计：1/8 尺度高斯（两次合成一次，方差相加 σ√2）再上采样。

    背景是低频量（阴影/渐变），1/8 尺度仍有 ~50 像素/周期的表达力；
    大核高斯在 1/8 尺度执行比 1/4 尺度快 ~4x。
    """
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, int(round(w * downscale))),
                              max(1, int(round(h * downscale)))),
                       interpolation=cv2.INTER_AREA)
    # 两次高斯合成一次（方差相加 σ√2）：内部区域严格等价（卷积结合律），
    # 省一半大核卷积时间
    sigma = max(1.0, max(small.shape) * sigma_ratio) * np.sqrt(2.0)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigma)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 1e-3, None)


def normalize_illumination(gray, strength=1.0):
    """gray: [0,1] float32。返回光照平坦化后的 [0,1] float32。"""
    if strength <= 0:
        return gray.copy()
    bg = estimate_background(gray)
    corrected = gray / bg
    lo, hi = corrected.min(), corrected.max()
    if hi - lo < 1e-6:
        return gray.copy()
    np.subtract(corrected, lo, out=corrected)
    corrected *= 1.0 / (hi - lo)
    if strength < 1.0:
        corrected *= strength
        corrected += (1.0 - strength) * gray
    np.clip(corrected, 0.0, 1.0, out=corrected)
    return corrected.astype(np.float32, copy=False)
