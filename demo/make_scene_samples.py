"""生成 5 种新场景的合成测试样本（场景自适应能力验证）。

样本清单：
1. sketch_photo.png   手机拍照手绘草图：铅笔弱线 + 径向阴影 + 纸纹噪声
2. document_scan.png  扫描文档：暗角 + 折痕阴影 + 文字行 + 污渍
3. blueprint.png      蓝图：深蓝底白线工程图
4. pixel_art.png      像素艺术：8bit 硬边块状图（无抗锯齿）
5. photo_like.png     类照片：平滑渐变 + 丰富色彩 + 结构边缘
"""
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "samples")
rng = np.random.default_rng(42)


def _paper_texture(h, w, strength=6):
    """纸张纤维纹理：低幅值高频噪声 + 少量斑点。"""
    noise = rng.normal(0, 1, (h, w)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.8)
    tex = noise * strength
    blotch = (rng.random((h // 6, w // 6)) < 0.01).astype(np.float32)
    blotch = cv2.resize(blotch, (w, h), interpolation=cv2.INTER_LINEAR)
    tex += blotch * -14.0
    return tex


def make_sketch_photo(w=640, h=480):
    """白纸 + 灰铅笔线（弱、渐变）+ 左上→右下阴影 + 纸纹。"""
    paper = np.full((h, w), 235.0, np.float32)
    paper += _paper_texture(h, w)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    shade = 1.0 - 0.28 * ((xx / w + yy / h) / 2.0) ** 1.2  # 0.72~1.0 渐变
    paper *= shade

    img = paper.copy()
    # 铅笔线：抗锯齿多段线（中心深度稳定=darkness，对标真实铅笔笔迹 0.4~0.5×背景）
    def stroke(p0, p1, p2, darkness=95, thick=2):
        canvas = np.zeros((h, w), np.float32)
        pts = []
        for t in np.linspace(0, 1, 160):
            x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0]
            y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
            pts.append([int(round(x)), int(round(y))])
        cv2.polylines(canvas, [np.array(pts, np.int32)], False, 1.0,
                      thickness=thick, lineType=cv2.LINE_AA)
        canvas = cv2.GaussianBlur(canvas, (0, 0), 0.6)  # 铅笔柔化
        img[:] = np.minimum(img, 255.0 - canvas * (255.0 - darkness))

    stroke((80, 100), (280, 60), (520, 180))
    stroke((90, 380), (300, 300), (560, 360))
    stroke((150, 220), (320, 240), (470, 120))
    stroke((200, 430), (350, 380), (580, 250), darkness=115, thick=2)
    cv2.circle(img, (170, 300), 55, 110, 2, cv2.LINE_AA)  # 弱圆

    return np.clip(img, 0, 255).astype(np.uint8)


def make_document_scan(w=560, h=720):
    """文档：装订/扫描阴影（左深右浅）+ 暗角 + 文字行 + 污渍 + 折痕。

    阴影强度对标真实扫描件（bgstd≈0.05+，弱于 0.03 的合成渐变图）。
    """
    img = np.full((h, w), 238.0, np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # 装订阴影：左侧深、向右渐亮（复印/扫描最常见的光照不均）
    img *= 1.0 - 0.26 * np.clip(1.0 - xx / (w * 0.55), 0, 1) ** 1.4
    # 暗角
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    img *= 1.0 - 0.18 * np.clip(r - 0.55, 0, 1) ** 1.5 * 2.2
    # 折痕阴影（斜向亮带边缘的暗线）
    img -= 26.0 * np.exp(-((xx * 0.8 + yy * 0.5 - 620) ** 2) / 900.0)
    img += _paper_texture(h, w, 5)
    img = np.clip(img, 0, 255)

    # 文字行（短横条模拟字符）
    dark = np.zeros((h, w), np.float32)
    for row in range(8):
        y = 70 + row * 76
        for cx in range(10):
            x = 60 + cx * 48
            if row == 3 and cx in (3, 4):
                continue  # 段落间隔
            for k in range(4):  # 每词 4 个“字符”
                gx = x + k * 10
                hh = rng.integers(6, 11)
                dark[y - hh:y, gx:gx + rng.integers(4, 8)] = 1.0
    dark = cv2.GaussianBlur(dark, (0, 0), 0.6)
    img = np.minimum(img, 255.0 - dark * 165.0)

    # 咖啡渍（环形浅棕渍 + 内部更浅）
    stain = np.zeros((h, w), np.float32)
    cv2.circle(stain, (430, 600), 55, 0.55, 6)
    cv2.circle(stain, (430, 600), 52, 0.30, -1)
    stain = cv2.GaussianBlur(stain, (0, 0), 1.2)
    img = img * (1.0 - stain * 0.35)

    return np.clip(img, 0, 255).astype(np.uint8)


def make_blueprint(w=640, h=480):
    """深蓝底白线工程图（蓝底 #1a3a6b 一类 + 白线 + 蓝图中常见的浅蓝网格）。"""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (107, 58, 26)  # BGR: 深蓝
    # 淡网格（浅蓝）
    for gx in range(0, w, 40):
        cv2.line(img, (gx, 0), (gx, h), (140, 85, 45), 1)
    for gy in range(0, h, 40):
        cv2.line(img, (0, gy), (w, gy), (140, 85, 45), 1)
    # 白色工程图元素
    cv2.rectangle(img, (80, 80), (560, 400), (235, 235, 235), 2)
    cv2.circle(img, (200, 240), 70, (245, 245, 245), 2)
    cv2.line(img, (300, 240), (520, 240), (235, 235, 235), 2)
    cv2.line(img, (410, 130), (410, 350), (235, 235, 235), 2)
    cv2.line(img, (300, 130), (520, 350), (245, 245, 245), 2)
    img = cv2.GaussianBlur(img, (0, 0), 0.5)
    return img


def make_pixel_art(w=320, h=240):
    """8-bit 风格像素图：块状硬边（放大 8x 的 40x30 逻辑图）。"""
    px, py = w // 8, h // 8
    canvas = np.zeros((py, px, 3), np.uint8)
    canvas[:] = (168, 220, 240)  # 浅色天空
    # 地面
    canvas[py - 6:, :] = (60, 140, 80)
    # 房子（黑轮廓块）
    canvas[10:22, 6:16] = (200, 180, 160)
    canvas[10:22, 6:16][0, :] = (0, 0, 0)
    for i in range(6):
        canvas[10 + i, 6 + i] = (0, 0, 0)
        canvas[10 + i, 15 - i] = (0, 0, 0)
    # 太阳
    canvas[3:6, 28:31] = (60, 230, 255)
    # 树
    canvas[14:24, 24:28] = (40, 90, 50)
    canvas[14, 25:27] = (0, 0, 0)
    return cv2.resize(canvas, (w, h), interpolation=cv2.INTER_NEAREST)


def make_photo_like(w=480, h=360):
    """类照片：平滑渐变天空 + 色彩丰富 + 结构边缘（山脊/建筑轮廓）。"""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    # 渐变天空（上深蓝下暖橙）
    t = (yy / h) ** 1.3
    img[:, :, 0] = 120 + 100 * t      # B
    img[:, :, 1] = 130 + 60 * t       # G
    img[:, :, 2] = 160 + 60 * (1 - t)  # R
    # 太阳（平滑亮斑）
    d = np.sqrt((xx - 340) ** 2 + (yy - 90) ** 2)
    glow = np.exp(-(d / 70.0) ** 2) * 90.0
    img += glow[..., None] * np.array([0.5, 0.7, 1.0])
    # 山（两层，结构边缘）
    ridge1 = h * 0.62 + 36 * np.sin(xx / 61.0) + 14 * np.sin(xx / 23.0 + 2)
    ridge2 = h * 0.74 + 30 * np.sin(xx / 44.0 + 1.2) + 10 * np.sin(xx / 17.0)
    m1 = yy > ridge1
    m2 = yy > ridge2
    img[m1] = img[m1] * 0.72 + np.array([25, 45, 60], np.float32)
    img[m2] = img[m2] * 0.55 + np.array([10, 15, 20], np.float32)
    # 细微噪声（照片感）
    img += rng.normal(0, 2.5, img.shape).astype(np.float32)
    img = cv2.GaussianBlur(img, (0, 0), 0.7)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    samples = {
        "sketch_photo.png": make_sketch_photo(),
        "document_scan.png": make_document_scan(),
        "blueprint.png": make_blueprint(),
        "pixel_art.png": make_pixel_art(),
        "photo_like.png": make_photo_like(),
    }
    for name, img in samples.items():
        path = os.path.join(OUT, name)
        cv2.imwrite(path, img)
        print(f"[OK] {path}  {img.shape[1]}x{img.shape[0]}")


if __name__ == "__main__":
    main()
