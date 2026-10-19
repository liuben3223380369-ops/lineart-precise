"""illumination 内容保护预滤波实验。

问题: 新 sigma(去 sqrt2) 在小图上让 bg 吃进暗线内容 -> test_circle_noisy
std_err 0.878 > 0.7。旧 sqrt2 靠过度平滑掩盖。
方案: 预滤波抹掉暗内容再高斯。对比:
  A 新sigma 无预滤波(现状, 已知失败)
  B 新sigma + dilate 3x3
  C 新sigma + close 3x3 (dilate+erode)
  D 旧 sqrt2 (基线)
指标: (1) 圆噪声测试 mean/std/cover (2) 大图 bg vs 全分辨率高斯 gt。
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tests"))
import cv2
import numpy as np

from test_precision import make_circle, radial_stats

K3 = np.ones((3, 3), np.uint8)


def bg_est(gray, downscale=0.125, sigma_ratio=0.035, legacy=False, pre=None):
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, round(w * downscale)),
                              max(1, round(h * downscale))),
                       interpolation=cv2.INTER_AREA)
    if pre == 'dilate':
        small = cv2.dilate(small, K3)
    elif pre == 'close':
        small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, K3)
    sig = max(1.0, max(small.shape) * sigma_ratio)
    if legacy:
        sig *= np.sqrt(2.0)
    bg_small = cv2.GaussianBlur(small, (0, 0), sig)
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)


def circle_case(tag, **kw):
    img = make_circle(noise=0.05)
    gray = img / 255.0
    bg = bg_est(gray, **kw)
    corrected = gray / np.clip(bg, 1e-3, None)
    lo, hi = corrected.min(), corrected.max()
    corrected = (corrected - lo) / (hi - lo)
    # 只看 bg 场本身在圆环附近的行为: 对比 bg 与"无圆纯背景"的偏差
    img_clean = make_circle(noise=0.0)
    gray_clean = img_clean / 255.0
    bg_clean = bg_est(gray_clean, **kw)
    ring = np.abs(gray_clean - 60.0 / 255.0) < 0.02  # 外部背景区
    leak = np.abs(bg - bg_clean)[ring].max()
    inside = np.hypot(*np.mgrid[0:192, 0:192].astype(np.float64)) < 30
    leak_in = np.abs(bg - bg_clean)[inside].max()
    print(f"{tag:28s} 外部bg泄漏={leak:.4f} 内部bg泄漏={leak_in:.4f}")


def main():
    circle_case('A 新sigma 无预滤波', pre=None)
    circle_case('B 新sigma + dilate3', pre='dilate')
    circle_case('C 新sigma + close3', pre='close')
    circle_case('D 旧sqrt2', legacy=True)

    # 大图 gt 一致性
    print()
    for name in ['cad_gray_bg', 'blueprint']:
        img = cv2.imread(f'demo/samples/{name}.png', cv2.IMREAD_GRAYSCALE)
        gray = img.astype(np.float32) / 255.0
        s = 2048 / max(gray.shape)
        big = cv2.resize(gray, (round(gray.shape[1] * s),
                                round(gray.shape[0] * s)),
                         interpolation=cv2.INTER_CUBIC)
        gt = cv2.GaussianBlur(big, (0, 0), max(1.0, max(big.shape) * 0.035))
        for tag, kw in [('A 新sigma', dict()),
                        ('B +dilate3', dict(pre='dilate')),
                        ('C +close3', dict(pre='close')),
                        ('D 旧sqrt2', dict(legacy=True))]:
            bg = bg_est(big, **kw)
            err = np.abs(bg - gt)
            print(f"{name} {tag:12s} max={err.max():.4f} "
                  f"rmse={np.sqrt((err**2).mean()):.5f}")


if __name__ == '__main__':
    main()
