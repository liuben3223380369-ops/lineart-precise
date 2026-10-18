"""生成演示与测试用合成图像：包含亚像素边缘、光照渐变、噪声等挑战场景。"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")


def subpixel_strip(size=160):
    """四条垂直边缘，真值位置 x = 40, 60.25, 80.5, 100.75。"""
    xs = np.arange(size, dtype=np.float64) + 0.5
    rows = np.arange(size)
    img = np.zeros((size, size))
    for edge_x, y0, y1 in ((40.0, 10, 70), (60.25, 80, 150),
                           (80.5, 10, 70), (100.75, 80, 150)):
        coverage = np.clip(edge_x - xs + 0.5, 0.0, 1.0) * 255.0
        band = (rows >= y0) & (rows < y1)
        img += np.where(band[:, None], coverage[None, :], 0.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def scene_shapes(size=480):
    """多形状场景：矩形 + 圆 + 三角 + 渐变背景 + 细线 + 轻噪声。"""
    img = np.zeros((size, size), np.float64)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img += 200.0 + 45.0 * (xx / size) * 0.5 + 35.0 * (yy / size) * 0.5
    img[(yy > 60) & (yy < 200) & (xx > 50) & (xx < 210)] = 90.0
    d = np.sqrt((yy - 150.0) ** 2 + (xx - 340.0) ** 2)
    img[d < 75.0] = 235.0
    tri = (yy > 300) & ((yy - 300) * 0.9 > np.abs(xx - 145))
    img[tri] = 60.0
    img[(np.abs(d - 75.0) < 1.2)] = 150.0
    for k in range(12):
        yline = 310 + 8 * k
        img[yline - 1:yline + 1, 280:440] = 120.0 + 8 * k
    rng = np.random.default_rng(11)
    img += rng.normal(0, 6.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def scene_circle(size=384):
    img = np.full((size, size), 215.0)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = np.sqrt((yy - 192.4) ** 2 + (xx - 191.6) ** 2)
    cov = np.clip(120.0 - d + 0.5, 0.0, 1.0)
    img = img * (1.0 - cov) + 70.0 * cov
    img += 25.0 * (xx / size)
    rng = np.random.default_rng(5)
    img += rng.normal(0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def scene_texture(size=224):
    """强纹理 + 弱结构场景：验证 FDoG 流线平滑与噪声自适应门限。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = 200.0 + 26.0 * np.sin(2 * np.pi * xx / 12.0) * np.sin(2 * np.pi * yy / 15.0)
    d = np.sqrt((yy - 112.0) ** 2 + (xx - 112.0) ** 2)
    cov = np.clip(60.0 - d + 0.5, 0, 1)
    img = img * (1 - cov) + 55.0 * cov
    rng = np.random.default_rng(21)
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = {
        "subpixel_strip.png": subpixel_strip,
        "shapes_scene.png": scene_shapes,
        "circle_scene.png": scene_circle,
        "texture_scene.png": scene_texture,
    }
    for name, fn in jobs.items():
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, fn())
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
