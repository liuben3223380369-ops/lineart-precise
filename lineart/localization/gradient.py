"""梯度算子：亚像素定位的输入场。

- gradient_sobel: 高斯平滑 + Sobel 3x3（基线）
- gradient_farid: Farid-Simon 5x5 最优导数滤波器，系统偏移从
  Sobel 的 -0.3px 降低到约 -0.05px
"""
import cv2
import numpy as np

# Farid-Simon 5x5 最优导数滤波器（平滑核 × 导数核可分离）
_FARID_SMOOTH = np.array([0.037037, 0.296296, 0.339506,
                          0.296296, 0.037037], dtype=np.float32)
# 导数核写成正导数（+∇I）形式：cv2.filter2D 系列是 correlation（不翻转
# 核），论文原始核 [+, +, 0, -, -] 在 correlation 下输出 -∇I，与 Sobel
# 的 +∇I 符号相反（v1.7.0 修正；曾致依赖方向语义的模块在 farid 下失效）
_FARID_DERIV = np.array([-0.037037, -0.296296, 0.0,
                         0.296296, 0.037037], dtype=np.float32)


def gradient_sobel(gray, sigma=1.2):
    """Sobel 梯度场。"""
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigma)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return mag, gx, gy


def gradient_farid(gray):
    """Farid-Simon 5x5 最优导数滤波器。

    比 Sobel 3x3 和 Scharr 更精确的方向导数估计，
    系统偏移从 -0.3px 降低到约 -0.05px。
    返回 (mag, gx, gy)。
    """
    gx = cv2.sepFilter2D(gray, cv2.CV_32F,
                         _FARID_DERIV, _FARID_SMOOTH)
    gy = cv2.sepFilter2D(gray, cv2.CV_32F,
                         _FARID_SMOOTH, _FARID_DERIV)
    mag = cv2.magnitude(gx, gy)
    return mag, gx, gy
