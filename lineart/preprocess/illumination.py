"""光照归一化：消除阴影与亮度渐变，保证不同光照下线条判定一致。"""
import cv2
import numpy as np


def estimate_background(gray, downscale=0.125, sigma_ratio=0.035,
                        min_sigma=2.0):
    """背景低频场估计：1/8 尺度高斯再上采样。

    背景是低频量（阴影/渐变），1/8 尺度仍有 ~50 像素/周期的表达力；
    大核高斯在 1/8 尺度执行比 1/4 尺度快 ~4x。
    v1.8.0: 修正有效 sigma —— 旧实现乘 sqrt(2)（"两次高斯合成一次"
    的历史遗留，但本函数只 blur 一次），导致等效平滑尺度偏大 sqrt(2)
    倍、大图与全分辨率大核高斯基准的偏差 max 0.032。去掉后大图偏差
    降至 0.002（阴影跟随更准）。
    同时引入绝对下限 min_sigma：小图（如 192px 输入 -> 1/8 尺度仅
    24px、比例 sigma 不足 1.0）上圆等大结构会泄漏进背景场、除法
    校正抹平内容对比度；实测 sigma>=1.5 后泄漏饱和消失，取 2.0
    留裕量（只影响 <=457px 输入，大图比例 sigma 远大于此）。
    """
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, int(round(w * downscale))),
                              max(1, int(round(h * downscale)))),
                       interpolation=cv2.INTER_AREA)
    sigma = max(min_sigma, max(small.shape) * sigma_ratio)
    bg_small = cv2.GaussianBlur(small, (0, 0), sigma)
    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return np.clip(bg, 1e-3, None)


def normalize_illumination(gray, strength=1.0):
    """gray: [0,1] float32。返回光照平坦化后的 [0,1] float32。

    v1.13.0: 入口防御 uint8 输入——strength<1 时 corrected([0,1]) 与
    gray([0,255]) 直接线性混合会把任意像素推过 1.0，clip 后整图全白
    （管线内 to_gray 恒为 float，此处仅防 API 误用）。
    """
    if strength <= 0:
        return gray.copy()
    if not np.issubdtype(gray.dtype, np.floating):
        gray = gray.astype(np.float32) * (1.0 / 255.0)
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
