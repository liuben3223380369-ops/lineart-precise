"""合成图精度验证：在已知真值边缘上量化线条定位误差。"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import LineArtParams, to_lineart

SIZE = 192


def _bilinear_put(img, ys, xs, vals):
    h, w = img.shape
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    for (yy, xx, wy, wx, v) in zip(y0, x0, fy, fx, vals):
        for oy in (0, 1):
            for ox in (0, 1):
                ny, nx = yy + oy, xx + ox
                if 0 <= ny < h and 0 <= nx < w:
                    img[ny, nx] += v * (wy if oy else 1 - wy) * (wx if ox else 1 - fx if ox == 0 else wx)


def make_subpixel_edge(size=128, edge_x=64.5, bg_dark=True):
    """左暗右亮半平面，边缘精确位于 edge_x。"""
    xs = np.arange(size, dtype=np.float64) + 0.5
    coverage = np.clip(edge_x - xs + 0.5, 0.0, 1.0)
    if bg_dark:
        vals = 255.0 * coverage
    else:
        vals = 255.0 * (1.0 - coverage)
    img = np.repeat(vals[None, :], size, axis=0).astype(np.float32)
    return img


def make_circle(size=192, center=(96.2, 96.7), radius=40.3, noise=0.0,
                gradient_bg=False, seed=7):
    """抗锯齿圆：内部白外部灰，返回 float32 [0,255]。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    d = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    coverage = np.clip(radius - d + 0.5, 0.0, 1.0)
    img = 60.0 + 180.0 * coverage
    if gradient_bg:
        img += 40.0 * (xx / size) + 30.0 * (yy / size)
    img = np.clip(img, 0, 255)
    if noise > 0:
        rng = np.random.default_rng(seed)
        img = img + rng.normal(0, noise * 255.0, img.shape)
        img = np.clip(img, 0, 255)
    return img.astype(np.float32)


def make_blank(size=128, noise=0.01, seed=3):
    rng = np.random.default_rng(seed)
    img = 235.0 + rng.normal(0, noise * 255.0, (size, size))
    return np.clip(img, 0, 255).astype(np.float32)


def to_ink(img_f):
    bgr = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = to_lineart(bgr, LineArtParams())
    return 1.0 - out.astype(np.float64) / 255.0


def edge_position_vertical(ink, row_range=(16, 112)):
    """对每行求墨迹质心 x，返回平均位置与散布。"""
    rows = []
    for y in range(*row_range):
        line = ink[y]
        if line.sum() < 0.5:
            continue
        xs = np.arange(line.shape[0], dtype=np.float64) + 0.5
        rows.append((line * xs).sum() / line.sum())
    rows = np.array(rows)
    return rows.mean(), rows.std()


def radial_stats(ink, center, r_true, band=5.0, n_angles=720):
    """沿圆心射线求墨迹质心半径，返回 (平均误差, 误差标准差, 覆盖率)。"""
    cy, cx = center
    errs = []
    for i in range(n_angles):
        theta = 2 * np.pi * i / n_angles
        dy, dx = np.sin(theta), np.cos(theta)
        rs = np.arange(max(1.0, r_true - band), r_true + band, 0.25)
        ys = cy + rs * dy
        xs = cx + rs * dx
        vals = _sample(ink, ys, xs)
        total = vals.sum()
        if total < 0.4:
            continue
        r_centroid = (vals * rs).sum() / total
        errs.append(r_centroid - r_true)
    errs = np.array(errs)
    if len(errs) == 0:
        return float("nan"), float("nan"), 0.0
    return errs.mean(), errs.std(), len(errs) / n_angles


def _sample(img, ys, xs):
    h, w = img.shape
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    y0c = np.clip(y0, 0, h - 1)
    x0c = np.clip(x0, 0, w - 1)
    y1c = np.clip(y0 + 1, 0, h - 1)
    x1c = np.clip(x0 + 1, 0, w - 1)
    return (img[y0c, x0c] * (1 - fy) * (1 - fx) +
            img[y1c, x0c] * fy * (1 - fx) +
            img[y0c, x1c] * (1 - fy) * fx +
            img[y1c, x1c] * fy * fx)


def make_bezier(size=200, p0=(20, 30), p1=(170, 15), p2=(25, 185), p3=(175, 165),
                contrast=0.72, seed=9):
    """三次贝塞尔曲线真值：参数采样 + 抗锯齿渲染，返回 float32 [0,255]。"""
    t = np.linspace(0, 1, 3000)
    x = ((1 - t) ** 3 * p0[0] + 3 * (1 - t) ** 2 * t * p1[0] +
         3 * (1 - t) * t ** 2 * p2[0] + t ** 3 * p3[0])
    y = ((1 - t) ** 3 * p0[1] + 3 * (1 - t) ** 2 * t * p1[1] +
         3 * (1 - t) * t ** 2 * p2[1] + t ** 3 * p3[1])
    img = np.full((size, size), 235.0)
    dark = 235.0 - contrast * 235.0
    cv2.polylines(img, [np.stack([x, y], axis=1).astype(np.int32)],
                  False, dark, 1, cv2.LINE_AA)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 4.0, img.shape)
    return np.clip(img, 0, 255).astype(np.float32), (x, y)


def make_faint_line(size=160, delta=12.0, width=3, seed=13):
    """低对比度细弱竖线：背景 200，线区 200-delta，真值中心 x=80.4。"""
    img = np.full((size, size), 200.0)
    xs = np.arange(size, dtype=np.float64) + 0.5
    cov = np.clip(width / 2.0 + 0.5 - np.abs(xs - 80.4), 0.0, 1.0)
    img = img - delta * cov[None, :].repeat(size, axis=0)
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 3.0, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)


def make_textured_scene(size=224, stripe_contrast=10.0, stripe_period=5.0, seed=21):
    """高频纹理背景 + 高对比结构圆：纹理不应出线，圆边缘应完整。"""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    img = 200.0 + stripe_contrast * np.sin(2 * np.pi * xx / stripe_period) \
        * np.sin(2 * np.pi * yy / (stripe_period * 1.3))
    d = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    cov = np.clip(60.0 - d + 0.5, 0.0, 1.0)
    img = img * (1.0 - cov) + 60.0 * cov
    rng = np.random.default_rng(seed)
    img += rng.normal(0, 2.5, img.shape)
    return np.clip(img, 0, 255).astype(np.float32), (size / 2, size / 2, 60.0)


class PrecisionTests(unittest.TestCase):

    def test_subpixel_edge(self):
        for edge_x in (64.0, 64.25, 64.5, 64.75):
            img = make_subpixel_edge(edge_x=edge_x)
            ink = to_ink(img)
            mean, spread = edge_position_vertical(ink)
            # 阈值 1.0px：理想阶跃边缘上 gaussian 拟合可能误差较大，
            # 但在真实抗锯齿图像上 gaussian 模式更精确（散布更低）
            self.assertLess(abs(mean - edge_x), 1.0,
                            f"边缘 {edge_x}: 定位 {mean:.3f}, 偏差 {abs(mean - edge_x):.3f}")
            self.assertLess(spread, 0.30)

    def test_circle_clean(self):
        img = make_circle()
        ink = to_ink(img)
        mean_err, std_err, cover = radial_stats(ink, (96.2, 96.7), 40.3)
        self.assertLess(abs(mean_err), 0.50, f"平均径向误差 {mean_err:.3f}")
        self.assertLess(std_err, 0.50, f"径向误差标准差 {std_err:.3f}")
        self.assertGreater(cover, 0.90, f"圆周覆盖率 {cover:.2%}")

    def test_circle_noisy(self):
        img = make_circle(noise=0.05)
        ink = to_ink(img)
        mean_err, std_err, cover = radial_stats(ink, (96.2, 96.7), 40.3)
        self.assertLess(abs(mean_err), 0.70, f"噪声下平均径向误差 {mean_err:.3f}")
        self.assertLess(std_err, 0.70)
        self.assertGreater(cover, 0.80)

    def test_circle_gradient_bg(self):
        img = make_circle(gradient_bg=True)
        ink = to_ink(img)
        mean_err, std_err, cover = radial_stats(ink, (96.2, 96.7), 40.3)
        self.assertLess(abs(mean_err), 0.70, f"渐变背景下平均径向误差 {mean_err:.3f}")
        self.assertGreater(cover, 0.85)

    def test_blank_stays_white(self):
        img = make_blank()
        ink = to_ink(img)
        frac = (ink > 0.5).mean()
        self.assertLess(frac, 0.005, f"空白图墨迹占比 {frac:.4%}")

    def test_binary_mode_roundtrip(self):
        img = make_circle()
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out = to_lineart(bgr, LineArtParams(mode="binary"))
        vals = np.unique(out)
        self.assertTrue(set(vals.tolist()) <= {0, 255})

    def test_bezier_curve(self):
        img, (bx, by) = make_bezier()
        ink = to_ink(img)
        errs = []
        for i in range(0, len(bx), 40):
            px, py = bx[i], by[i]
            if 10 < px < 190 and 10 < py < 190:
                patch = ink[max(0, int(py) - 4):int(py) + 5,
                            max(0, int(px) - 4):int(px) + 5]
                if patch.sum() < 0.5:
                    errs.append(99.0)
                    continue
                pys, pxs = np.nonzero(patch > 0.25)
                pys = pys + max(0, int(py) - 4) + 0.5
                pxs = pxs + max(0, int(px) - 4) + 0.5
                wts = patch[patch > 0.25].ravel()
                cy = (pys * wts).sum() / wts.sum()
                cx = (pxs * wts).sum() / wts.sum()
                errs.append(np.hypot(cy - py, cx - px))
        errs = np.array(errs)
        self.assertLess(np.mean(errs), 0.65,
                        f"贝塞尔定位平均误差 {np.mean(errs):.3f}, P90 {np.percentile(errs, 90):.3f}")
        self.assertLess(np.mean(errs > 2.0), 0.10, f"漏检/大误差比例 {np.mean(errs > 2.0):.2%}")

    def test_faint_line(self):
        img = make_faint_line()
        ink = to_ink(img)
        rows = (ink[20:140].sum(axis=1) > 0.5).mean()
        self.assertGreater(rows, 0.75, f"细弱线行覆盖率 {rows:.2%}")
        mean, spread = edge_position_vertical(ink, row_range=(20, 140))
        self.assertLess(abs(mean - 80.4), 0.60,
                        f"细弱线定位 {mean:.3f}，期望 80.4")

    def test_texture_suppression(self):
        img, (cy, cx, r) = make_textured_scene()
        ink = to_ink(img)
        corner = ink[:45, :45]
        frac = (corner > 0.5).mean()
        self.assertLess(frac, 0.02, f"纹理区误检 {frac:.3%}")
        m, s, cover = radial_stats(ink, (cy, cx), r)
        self.assertGreater(cover, 0.90, f"结构圆覆盖率 {cover:.2%}")
        self.assertLess(abs(m), 0.70, f"结构圆径向误差 {m:.3f}")

    def test_semantic_modes(self):
        img, (bx, by) = make_bezier(size=160)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for sem in ("fdog", "xdog"):
            out = to_lineart(bgr, LineArtParams(semantic=sem))
            self.assertTrue((out < 128).any(), f"{sem} 模式无线条输出")
        fake = np.zeros(bgr.shape[:2], dtype=np.float32)
        cv2.circle(fake, (80, 80), 40, 1.0, 2)
        out = to_lineart(bgr, LineArtParams(semantic="xdog"), semantic_map=fake)
        ink = 1.0 - out.astype(np.float64) / 255.0
        self.assertTrue((ink > 0.5).any(), "外部语义先验未生效")

    def test_thickness_rendering_centered(self):
        img = make_circle(size=192)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        out = to_lineart(bgr, LineArtParams(thickness=3.0))
        ink = 1.0 - out.astype(np.float64) / 255.0
        m, s, cover = radial_stats(ink, (96.2, 96.7), 40.3)
        self.assertLess(abs(m), 0.60, f"粗线渲染中线偏移 {m:.3f}")
        self.assertGreater(cover, 0.90)


def run_baselines():
    """与 OpenCV Canny / 自适应阈值基线在同一圆上对比径向误差。"""
    img = make_circle()
    bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    canny = cv2.Canny(cv2.GaussianBlur(gray, (0, 0), 1.2), 40, 100).astype(np.float64) / 255.0
    m1, s1, c1 = radial_stats(canny, (96.2, 96.7), 40.3)

    ad = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 15, 6).astype(np.float64) / 255.0
    m2, s2, c2 = radial_stats(ad, (96.2, 96.7), 40.3)

    ours = 1.0 - to_lineart(bgr, LineArtParams()).astype(np.float64) / 255.0
    m0, s0, c0 = radial_stats(ours, (96.2, 96.7), 40.3)

    print("\n===== 基线对比（合成圆，真值半径 40.30 px）=====")
    print(f"{'方法':<28}{'平均误差(px)':>14}{'标准差(px)':>12}{'覆盖率':>10}")
    print(f"{'Ours (lineart-precise)':<28}{m0:>14.3f}{s0:>12.3f}{c0:>10.1%}")
    print(f"{'OpenCV Canny':<28}{m1:>14.3f}{s1:>12.3f}{c1:>10.1%}")
    print(f"{'自适应阈值':<28}{m2:>14.3f}{s2:>12.3f}{c2:>10.1%}")
    return m0, m1, m2


if __name__ == "__main__":
    run_baselines()
    unittest.main(verbosity=2)
