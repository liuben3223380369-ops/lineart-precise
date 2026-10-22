#!/usr/bin/env python3
"""多风格合成测试图生成（v1.27.2 本地模型调试）。

生成 4 类风格图 + 线条真值 mask（*_gt.png，白=线）：
- ink_guofeng.png   古风水墨：宣纸底 + 水墨晕染 + 毛笔变宽笔触 + 远山层叠
- pencil_sketch.png 素描：纸纹 + 铅笔排线阴影 + 软边轮廓 + 炭笔深线
- anime_manga.png   二次元漫画：大色块 + 粗硬轮廓 + 速度线 + 网点纸
- landscape_photo.png 风景照：天空渐变 + 远山剪影 + 水面波光 + 树簇

真实风格照片（人像/建筑/静物等 12 张）从 OpenCV 样本库下载于
demo/bench/，无真值——用结构参考指标评测。
"""
import cv2
import numpy as np

W, H = 1280, 960
rng = np.random.RandomState(11)


def paper_texture(base, fiber=7, amp=4):
    """纸张纤维纹理（低幅值噪声 + 轻模糊）。kernel 取奇数。"""
    if fiber % 2 == 0:
        fiber += 1
    noise = rng.randint(-amp, amp + 1, (H, W, 1)).astype(np.int16)
    out = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    out = cv2.GaussianBlur(out, (fiber, fiber), 0)
    return out


def brush_stroke(img, gt, p0, p1, width, color, alpha=1.0, blur=1.5):
    """毛笔笔触：变宽度线段 + 端部圆头 + 轻模糊（边缘不规则）。"""
    ctrl = ((p0[0] + p1[0]) // 2 + rng.randint(-80, 80),
            (p0[1] + p1[1]) // 2 + rng.randint(-60, 60))
    pts = []
    for t in np.linspace(0, 1, 40):
        # 二次贝塞尔
        x = (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * ctrl[0] + t ** 2 * p1[0]
        y = (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * ctrl[1] + t ** 2 * p1[1]
        # 宽度沿笔触正弦变化（起笔收笔细）
        w = width * (0.45 + 0.55 * np.sin(np.pi * t))
        pts.append((int(x), int(y), max(1, int(w))))
    layer = img.copy()
    gtl = np.zeros((H, W), np.uint8)
    for (x, y, w) in pts:
        cv2.circle(layer, (x, y), w // 2, color, -1, cv2.LINE_AA)
        cv2.circle(gtl, (x, y), max(1, w // 2), 255, -1)
    # 笔触区域 = layer 与原图有实质差异的像素（半径 blur 内的小晕染保留）
    diff = np.abs(layer.astype(np.int32) - img.astype(np.int32)).sum(2)
    mask = diff > 10
    if blur > 0:
        layer = cv2.GaussianBlur(layer, (0, 0), blur)
    if alpha < 1.0:
        layer = cv2.addWeighted(img, 1 - alpha, layer, alpha, 0)
    img[mask] = layer[mask]
    gt[mask | (gtl > 0)] = 255
    return img, gt


def gen_guofeng():
    """古风水墨：宣纸底 + 水墨晕染 + 笔触 + 远山 + 印章。"""
    base = np.full((H, W, 3), 232, np.uint8)
    base[:, :, 0] = np.clip(base[:, :, 0].astype(int) + 10, 0, 255)  # 米黄
    base = paper_texture(base, amp=3)
    img = base.copy()
    gt = np.zeros((H, W), np.uint8)
    # 远山层叠（低对比大形态，由浅到深）
    for i, (ybase, contrast) in enumerate([(430, 200), (500, 185), (570, 165)]):
        pts = [(0, ybase + rng.randint(-30, 30))]
        x = 0
        while x < W:
            x += rng.randint(120, 220)
            pts.append((x, ybase + rng.randint(-80, 80)))
        layer = base.copy()
        cv2.polylines(layer, [np.array(pts)], False, (contrast,) * 3, 2,
                      cv2.LINE_AA)
        poly = np.array(pts + [(W, H), (0, H)])
        cv2.fillPoly(layer, [poly], (int(contrast + 12),) * 3)
        img = cv2.addWeighted(img, 0.75 if i else 0.9, layer, 0.25 if i else 0.1, 0)
    # 水墨晕染（大半径低对比）
    for _ in range(6):
        x, y = rng.randint(150, W - 150), rng.randint(150, H - 350)
        r = rng.randint(90, 220)
        blob = np.full((H, W, 3), 235, np.uint8)
        cv2.circle(blob, (x, y), r, (208,) * 3, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), r / 3)
        img = np.minimum(img, blob)
    # 主笔触（树枝风格：主干 + 分枝）
    img, gt = brush_stroke(img, gt, (180, 780), (560, 420), 7, (58, 55, 52))
    img, gt = brush_stroke(img, gt, (560, 420), (760, 300), 4, (75, 72, 68))
    img, gt = brush_stroke(img, gt, (560, 420), (480, 250), 3, (85, 82, 78))
    img, gt = brush_stroke(img, gt, (760, 300), (980, 360), 3, (95, 92, 88))
    # 地面苔点
    for _ in range(24):
        x, y = rng.randint(200, W - 200), rng.randint(760, H - 80)
        cv2.circle(img, (x, y), rng.randint(2, 5), (90 + rng.randint(40),) * 3, -1)
        cv2.circle(gt, (x, y), 3, 255, -1)
    # 印章
    cv2.rectangle(img, (W - 140, H - 140), (W - 60, H - 60), (60, 40, 160), -1)
    cv2.rectangle(gt, (W - 140, H - 140), (W - 60, H - 60), 255, -1)
    return img, gt


def gen_sketch():
    """铅笔素描：排线阴影 + 软边轮廓 + 炭笔深线 + 涂抹区。"""
    base = paper_texture(np.full((H, W, 3), 238, np.uint8), amp=5)
    img = base.copy()
    gt = np.zeros((H, W), np.uint8)
    # 主体软边轮廓（鸡蛋 + 罐子形体）
    for (cx, cy, r, shade) in [(420, 480, 190, 190), (860, 520, 230, 175)]:
        cv2.circle(img, (cx, cy), r, (shade,) * 3, 3, cv2.LINE_AA)
        cv2.circle(gt, (cx, cy), r, 255, 2)
    cv2.ellipse(img, (860, 560), (230, 260), 0, 200, 340, (150,) * 3, 3,
                cv2.LINE_AA)
    cv2.ellipse(gt, (860, 560), (230, 260), 0, 200, 340, 255, 2)
    # 排线阴影（平行短线簇，垂直于光照方向）
    for (x0, y0, x1, y1, step, angle, shade) in (
            (250, 700, 580, 700, 6, 62, 165), (700, 780, 1050, 780, 7, 118, 150)):
        n = (x1 - x0) // step
        for i in range(n):
            x = x0 + i * step + rng.randint(-2, 3)
            ln = rng.randint(60, 130)
            dx = int(ln * np.cos(np.radians(angle)))
            dy = int(ln * np.sin(np.radians(angle)))
            cv2.line(img, (x, y0), (x + dx, y0 + dy),
                     (shade + rng.randint(15),) * 3, 1, cv2.LINE_AA)
            cv2.line(gt, (x, y0), (x + dx, y0 + dy), 255, 1)
    # 炭笔深线（关键结构）
    for (p0, p1) in [((300, 250), (560, 340)), ((560, 340), (900, 280)),
                     ((350, 850), (950, 880))]:
        cv2.line(img, p0, p1, (70 + rng.randint(30),) * 3, 2, cv2.LINE_AA)
        cv2.line(gt, p0, p1, 255, 2)
    # 手指涂抹区（模糊渐变阴影）
    smear = cv2.GaussianBlur(img, (0, 0), 9)
    m = np.zeros((H, W), np.float32)
    cv2.circle(m, (640, 640), 260, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), 120)[:, :, None]
    img = (img * (1 - 0.5 * m) + smear * 0.5 * m).astype(np.uint8)
    return img, gt


def gen_anime():
    """二次元漫画：大色块 + 粗硬轮廓 + 速度线 + 网点纸。"""
    img = np.full((H, W, 3), 250, np.uint8)
    gt = np.zeros((H, W), np.uint8)
    # 背景大色块（天空/地面渐变）
    img[:H // 2] = (200, 225, 245)
    img[H // 2:] = (235, 210, 180)
    # 人物大色块（简化人形：头/身/发）
    cv2.circle(img, (560, 330), 150, (252, 238, 225), -1)           # 脸
    hair = np.zeros((H, W), np.uint8)
    cv2.circle(hair, (560, 300), 175, 255, -1)
    hair[:300] = 0
    img[hair > 0] = (70, 55, 95)                                    # 头发
    cv2.rectangle(img, (430, 480), (690, 900), (240, 240, 245), -1)  # 身体
    cv2.rectangle(img, (640, 480), (900, 760), (200, 90, 90), -1)    # 外套
    # 轮廓粗线（硬边）
    for (p0, p1, w) in [((385, 330), (385, 620), 5), ((735, 330), (735, 620), 5),
                        ((430, 470), (690, 470), 4), ((560, 180), (560, 60), 6)]:
        cv2.line(img, p0, p1, (25,) * 3, w, cv2.LINE_AA)
        cv2.line(gt, p0, p1, 255, w)
    cv2.ellipse(img, (560, 330), (150, 170), 0, 180, 360, (25,) * 3, 5,
                cv2.LINE_AA)
    cv2.ellipse(gt, (560, 330), (150, 170), 0, 180, 360, 255, 4)
    cv2.ellipse(img, (560, 300), (175, 180), 0, 170, 360, (25,) * 3, 6)
    cv2.ellipse(gt, (560, 300), (172, 178), 0, 170, 360, 255, 5)
    # 五官（细线）
    cv2.ellipse(img, (505, 320), (28, 16), 0, 0, 360, (30,) * 3, 3, cv2.LINE_AA)
    cv2.ellipse(gt, (505, 320), (28, 16), 0, 0, 360, 255, 2)
    cv2.ellipse(img, (615, 320), (28, 16), 0, 0, 360, (30,) * 3, 3, cv2.LINE_AA)
    cv2.ellipse(gt, (615, 320), (28, 16), 0, 0, 360, 255, 2)
    cv2.line(img, (560, 345), (560, 390), (30,) * 3, 3, cv2.LINE_AA)
    cv2.line(gt, (560, 345), (560, 390), 255, 2)
    # 速度线（左上辐射）
    for ang in np.linspace(200, 250, 9):
        dx, dy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        p0 = (int(560 + dx * 320), int(500 + dy * 320))
        p1 = (int(560 + dx * 720), int(500 + dy * 720))
        cv2.line(img, p0, p1, (60,) * 3, 3, cv2.LINE_AA)
        cv2.line(gt, p0, p1, 255, 2)
    # 网点纸区域（右侧背景 dot pattern）
    ys, xs = np.mgrid[0:H, 0:W]
    dot_zone = (xs > 950) & (ys < 500)
    dot = ((xs % 8 == 0) & (ys % 8 == 0)).astype(np.uint8)
    img[dot_zone & (dot > 0)] = 120
    gt[dot_zone & (dot > 0)] = 255
    return img, gt


def gen_landscape():
    """风景照片风格：天空渐变 + 远山剪影 + 水面波光 + 树簇。"""
    img = np.zeros((H, W, 3), np.uint8)
    # 天空渐变
    sky = np.linspace(235, 175, H).astype(np.uint8)
    img[:, :, 0] = sky[:, None]
    img[:, :, 1] = np.clip(sky[:, None].astype(int) - 8, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(sky[:, None].astype(int) - 25, 0, 255).astype(np.uint8)
    # 太阳光晕
    glow = np.zeros((H, W), np.float32)
    cv2.circle(glow, (980, 170), 90, 1.0, -1)
    glow = cv2.GaussianBlur(glow, (0, 0), 60)[:, :, None]
    img = np.clip(img.astype(np.float32) + glow * 45, 0, 255).astype(np.uint8)
    gt = np.zeros((H, W), np.uint8)
    # 云彩软团
    for _ in range(5):
        x, y = rng.randint(80, W - 80), rng.randint(60, 300)
        blob = np.full((H, W, 3), 999, np.uint16)  # max 起效
        cv2.circle(blob, (x, y), rng.randint(50, 110), (245,) * 3, -1)
        blob = np.minimum(blob, 255).astype(np.uint8)
        blob = cv2.GaussianBlur(blob, (0, 0), 25)
        img = cv2.max(img, blob)
    # 远山三层剪影（低对比渐进）
    for i, (ybase, depth) in enumerate([(560, 195), (600, 165), (650, 130)]):
        pts = [(0, ybase + 40)]
        x = 0
        while x < W:
            x += rng.randint(150, 280)
            pts.append((x, ybase + rng.randint(-90, 60)))
        cv2.fillPoly(img, [np.array(pts + [(W, H), (0, H)])], (depth,) * 3)
        cv2.polylines(img, [np.array(pts)], False, (int(depth * 0.55),) * 3, 2,
                      cv2.LINE_AA)
        cv2.polylines(gt, [np.array(pts)], False, 255, 2)
    # 水面（下半）：渐变 + 水平波光线
    img[H // 2 + 140:] = np.clip(img[H // 2 + 140:].astype(int) - 20, 0,
                                 255).astype(np.uint8)
    for _ in range(90):
        y = rng.randint(H // 2 + 150, H - 30)
        x = rng.randint(60, W - 200)
        ln = rng.randint(50, 190)
        shade = rng.randint(160, 225)
        cv2.line(img, (x, y), (x + ln, y), (shade,) * 3, 1, cv2.LINE_AA)
        cv2.line(gt, (x, y), (x + ln, y), 255, 1)
    # 前景树簇（分形短线）
    def tree(x0, y0, size, depth):
        if depth == 0 or size < 8:
            return
        ang = rng.uniform(-0.55, 0.55) - np.pi / 2
        ln = size * rng.uniform(0.7, 1.0)
        x1 = int(x0 + np.cos(ang) * ln)
        y1 = int(y0 + np.sin(ang) * ln)
        w = max(1, int(depth))
        cv2.line(img, (x0, y0), (x1, y1), (55 + rng.randint(35),) * 3, w,
                 cv2.LINE_AA)
        cv2.line(gt, (x0, y0), (x1, y1), 255, max(1, w - 1))
        tree(x1, y1, size * 0.72, depth - 1)
        tree(x1, y1, size * 0.66, depth - 1)
    tree(210, H - 60, 150, 6)
    tree(1080, H - 50, 120, 5)
    # 轻噪点（照片感）
    noise = rng.randint(-7, 8, (H, W, 1)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img, gt


def main():
    import os
    os.chdir(os.path.join(os.path.dirname(__file__), "bench"))
    for name, fn in (("ink_guofeng", gen_guofeng),
                     ("pencil_sketch", gen_sketch),
                     ("anime_manga", gen_anime),
                     ("landscape_syn", gen_landscape)):
        img, gt = fn()
        cv2.imwrite(f"{name}.png", img)
        cv2.imwrite(f"{name}_gt.png", gt)
        cov = (gt > 0).mean()
        print(f"{name}.png: {W}x{H} 线条覆盖率 {cov:.1%}")


if __name__ == "__main__":
    main()
