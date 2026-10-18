"""背景上采样方案对比：CUBIC 直采 vs 级联 2x。

gt = 全分辨率大核高斯（sigma = 0.035 * max(h,w)）。
指标: max err（线条附近振铃最敏感）/ RMSE / 耗时。
"""
import time
import cv2
import numpy as np

def bg_downscale(gray, downscale=0.125, sigma_ratio=0.035):
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (max(1, int(round(w * downscale))),
                              max(1, int(round(h * downscale)))),
                       interpolation=cv2.INTER_AREA)
    sigma = max(1.0, max(small.shape) * sigma_ratio) * np.sqrt(2.0)
    return cv2.GaussianBlur(small, (0, 0), sigma)


def up_cubic(bg_small, w, h):
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)


def up_cascade(bg_small, w, h):
    cur = bg_small
    cw, ch = cur.shape[1], cur.shape[0]
    while cw < w or ch < h:
        nw, nh = min(w, cw * 2), min(h, ch * 2)
        cur = cv2.resize(cur, (nw, nh), interpolation=cv2.INTER_LINEAR)
        cw, ch = nw, nh
    return cur


def up_pyrup(bg_small, w, h):
    cur = bg_small
    while cur.shape[1] < w or cur.shape[0] < h:
        cur = cv2.pyrUp(cur)  # 尺寸 2n+1，最后统一回目标
    return cv2.resize(cur, (w, h), interpolation=cv2.INTER_LINEAR)


def bench(name, gray):
    h, w = gray.shape[:2]
    t0 = time.perf_counter()
    gt = cv2.GaussianBlur(gray, (0, 0), max(1.0, max(h, w) * 0.035))
    t_gt = time.perf_counter() - t0

    small = bg_downscale(gray)
    for tag, fn in [('CUBIC', up_cubic), ('cascade', up_cascade),
                    ('pyrUp', up_pyrup)]:
        t0 = time.perf_counter()
        bg = fn(small, w, h)
        dt = time.perf_counter() - t0
        err = np.abs(bg - gt)
        print(f"{name} {tag:8s} max={err.max():.4f} rmse={np.sqrt((err**2).mean()):.5f} t={dt*1000:.1f}ms")
    print(f"{name} gt      t={t_gt*1000:.1f}ms  shape={gray.shape}")


def main():
    for name in ['cad_gray_bg', 'blueprint']:
        img = cv2.imread(f'demo/samples/{name}.png', cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
        s = 2048 / max(img.shape)
        big = cv2.resize(img, (int(round(img.shape[1] * s)),
                               int(round(img.shape[0] * s))),
                         interpolation=cv2.INTER_CUBIC)
        bench(name, big)


if __name__ == '__main__':
    main()
