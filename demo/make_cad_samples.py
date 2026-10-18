"""生成 CAD 场景合成测试图像。

场景：
1. 黑底多色线（CAD 典型截图）
2. 黑底 + 网格背景
3. 黑底 + 文字标注
4. 粗线条（轮廓线）
5. 灰底（扫描图纸）
"""
import os

import cv2
import numpy as np

OUT_DIR = os.path.join(os.path.dirname(__file__), "samples")


def cad_black_color(size=512):
    """黑底多色线 CAD 截图。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    colors = {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "cyan": (255, 255, 0),
        "blue": (255, 0, 0),
        "magenta": (255, 0, 255),
        "white": (255, 255, 255),
    }
    for i, (name, color) in enumerate(colors.items()):
        y = 40 + i * 60
        cv2.line(img, (30, y), (200, y), color, 1, cv2.LINE_AA)
        cv2.putText(img, name, (220, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)
    cv2.rectangle(img, (30, 430), (250, 480), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (380, 150), 80, (0, 200, 255), 1, cv2.LINE_AA)
    cv2.circle(img, (380, 150), 40, (0, 0, 255), 2, cv2.LINE_AA)
    for i in range(6):
        angle = i * 60 * np.pi / 180
        x2 = 380 + 80 * np.cos(angle)
        y2 = 150 + 80 * np.sin(angle)
        cv2.line(img, (380, 150), (int(x2), int(y2)), (0, 255, 255), 1, cv2.LINE_AA)
    return img


def cad_grid_bg(size=512, grid_period=8):
    """黑底 + 网格背景 + 彩色线条。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for x in range(0, size, grid_period):
        cv2.line(img, (x, 0), (x, size), (30, 30, 30), 1)
    for y in range(0, size, grid_period):
        cv2.line(img, (0, y), (size, y), (30, 30, 30), 1)
    cv2.rectangle(img, (60, 60), (200, 200), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (260, 60), (450, 200), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.circle(img, (350, 380), 70, (0, 255, 0), 1, cv2.LINE_AA)
    for i in range(10):
        y = 300 + i * 18
        cv2.line(img, (50, y), (180, y), (255, 0, 0), 1, cv2.LINE_AA)
    return img


def cad_with_text(size=512):
    """黑底 + 文字标注 + 线条。"""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.line(img, (50, 100), (450, 100), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (50, 200), (450, 200), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (100, 50), (100, 450), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(img, (300, 50), (300, 450), (255, 255, 255), 1, cv2.LINE_AA)
    labels = ["DIM A", "DIM B", "DIM C", "120.5", "80.25", "R10"]
    positions = [(240, 90), (240, 190), (90, 260), (290, 260), (180, 350), (320, 350)]
    for label, pos in zip(labels, positions):
        cv2.putText(img, label, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def cad_thick_lines(size=512):
    """粗线条 CAD（轮廓线）。"""
    img = np.ones((size, size, 3), dtype=np.uint8) * 240
    cv2.rectangle(img, (50, 50), (200, 200), (0, 0, 0), 4, cv2.LINE_AA)
    cv2.rectangle(img, (250, 50), (450, 200), (0, 0, 0), 3, cv2.LINE_AA)
    cv2.circle(img, (350, 380), 80, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.line(img, (50, 300), (200, 450), (0, 0, 0), 5, cv2.LINE_AA)
    for i in range(8):
        y = 100 + i * 35
        cv2.line(img, (280, y), (480, y), (80, 80, 80), 2, cv2.LINE_AA)
    return img


def cad_gray_bg(size=512):
    """灰底扫描图纸。"""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(7)
    img = img + rng.normal(0, 8, img.shape).astype(np.int32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.rectangle(img, (40, 40), (220, 220), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.rectangle(img, (260, 40), (470, 220), (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(img, (350, 380), 90, (0, 0, 0), 1, cv2.LINE_AA)
    for i in range(12):
        y = 80 + i * 30
        cv2.line(img, (30, y), (230, y), (50, 50, 50), 1, cv2.LINE_AA)
    return img


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    jobs = {
        "cad_black_color.png": cad_black_color,
        "cad_grid_bg.png": cad_grid_bg,
        "cad_with_text.png": cad_with_text,
        "cad_thick_lines.png": cad_thick_lines,
        "cad_gray_bg.png": cad_gray_bg,
    }
    for name, fn in jobs.items():
        path = os.path.join(OUT_DIR, name)
        cv2.imwrite(path, fn())
        print(f"[OK] {path}")


if __name__ == "__main__":
    main()
