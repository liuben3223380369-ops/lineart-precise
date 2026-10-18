"""场景自适应转换测试：检测标签 + 各场景转换质量。"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import LineArtParams, to_lineart
from lineart.semantics.adaptive import adaptive_semantic, sauvola_threshold
from lineart.scenes import detect_scene, scene_info_for
from lineart.pipeline import _apply_scene

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "demo", "samples")


def _load(name):
    return cv2.imread(os.path.join(SAMPLES, name))


def _gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0


class SceneDetectionTests(unittest.TestCase):
    CASES = {
        "blueprint.png": "blueprint",
        "sketch_photo.png": "sketch",
        "document_scan.png": "sketch",
        "pixel_art.png": "pixel",
        "photo_like.png": "photo",
        "circle_scene.png": "lineart",
        "shapes_scene.png": "lineart",
        "texture_scene.png": "lineart",
    }

    def test_detect_all(self):
        for name, expected in self.CASES.items():
            with self.subTest(sample=name):
                img = _load(name)
                self.assertIsNotNone(img, f"缺少样本 {name}")
                info = detect_scene(img)
                self.assertEqual(info.scene, expected,
                                 f"{name}: {info.scene} != {expected}")


class AdaptiveSemanticTests(unittest.TestCase):
    def test_sauvola_uniform_region(self):
        """均匀区域阈值应低于均值（Sauvola 的 k 项），避免平坦背景出噪。"""
        gray = np.full((64, 64), 0.9, np.float32)
        t, m, s = sauvola_threshold(gray, window=15, k=0.24)
        self.assertTrue((t <= m + 1e-6).all())
        self.assertLess(float(t.std()), 1e-6)

    def test_shadow_consistency(self):
        """同一圆在亮区与阴影区的 Sauvola 响应强度一致（全局阈值做不到）。"""
        h, w = 200, 200
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        paper = 0.92 - 0.42 * (xx / w)  # 左亮(0.92)右暗(0.50)：阴影渐变明显
        img = paper.copy()
        d1 = np.sqrt((yy - 100) ** 2 + (xx - 55) ** 2)
        d2 = np.sqrt((yy - 100) ** 2 + (xx - 145) ** 2)
        line = (np.abs(d1 - 40) < 1.5) | (np.abs(d2 - 40) < 1.5)
        img[line] = 0.30  # 深度统一（约为暗区背景的 60%）

        # 全局阈值：暗区背景(<0.55)被误判为墨
        global_ink = (img < 0.55).astype(np.float32)
        dark_zone = (xx > 150)
        bg_dark = float(global_ink[(~line) & dark_zone].mean())
        self.assertGreater(bg_dark, 0.4, "全局阈值应在暗区背景误响应（对照实验失效）")

        # Sauvola 在两个区域都稳定
        ink = adaptive_semantic(img, sensitivity=0.5)
        bright_zone = (xx < 50)
        r_bright = float(ink[line & bright_zone].mean())
        r_dark = float(ink[line & dark_zone].mean())
        self.assertGreater(r_bright, 0.7)
        self.assertGreater(r_dark, 0.7)
        self.assertLess(abs(r_bright - r_dark), 0.25)
        bg_leak = float(ink[(~line) & dark_zone].mean())
        self.assertLess(bg_leak, 0.15, "Sauvola 在暗区背景泄漏")


class ConversionTests(unittest.TestCase):
    def test_sketch_photo_convert(self):
        """拍照草图：阴影下弱线保留、背景干净。"""
        img = _load("sketch_photo.png")
        out = to_lineart(img, LineArtParams())
        ink = 1.0 - out.astype(np.float64) / 255.0
        h, w = ink.shape
        # 左上（最亮）与右下（最暗）阴影区都应保留线条
        lu = ink[:h // 2, :w // 2]
        rd = ink[h // 2:, w // 2:]
        self.assertGreater(float(lu.mean()), 0.01, "亮区线条丢失")
        self.assertGreater(float(rd.mean()), 0.01, "阴影区线条丢失")
        # 背景四角应干净（无阴影残留墨迹）
        corners = [ink[:40, :40], ink[:40, -40:], ink[-40:, :40], ink[-40:, -40:]]
        for c in corners:
            self.assertLess(float(c.mean()), 0.08, "角落背景残留阴影墨迹")

    def test_document_convert(self):
        """扫描文档：文字行保留。"""
        img = _load("document_scan.png")
        out = to_lineart(img, LineArtParams())
        ink = 1.0 - out.astype(np.float64) / 255.0
        # 输入文字行位于 y=70+row*76 附近；输出在这些行应有墨迹
        src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rows_src = [70 + r * 76 for r in range(8)]
        kept = 0
        for y in rows_src:
            band_src = src_gray[y - 12:y + 4].mean()
            band_out = ink[max(0, y - 14):y + 6].max()
            if band_src < 200:  # 该行确有文字
                kept += 1 if band_out > 0.25 else 0
        self.assertGreaterEqual(kept, 6, f"8 行文字仅保留 {kept} 行")

    def test_blueprint_convert(self):
        """蓝图：转成白底黑线，无蓝色残留。"""
        img = _load("blueprint.png")
        out = to_lineart(img, LineArtParams())
        self.assertEqual(out.shape[:2], img.shape[:2])
        if out.ndim == 3:
            b = out[:, :, 0].astype(int)
            r = out[:, :, 2].astype(int)
            self.assertLess(float(np.abs(b - r).mean()), 12.0, "输出残留蓝色")
            gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        else:
            gray_out = out
        ink = 1.0 - gray_out.astype(np.float64) / 255.0
        # 白色工程线（圆/矩形/对角线）应转为黑线
        self.assertGreater(float(ink.mean()), 0.01, "蓝图白线丢失")
        # 输出底应为白：中央空旷区干净
        self.assertLess(float(ink[195:245, 195:245].mean()), 0.10,
                        "蓝图底色未转白")

    def test_pixel_art_convert(self):
        """像素艺术：输出尺寸与输入一致，结构保留。"""
        img = _load("pixel_art.png")
        out = to_lineart(img, LineArtParams())
        self.assertEqual(out.shape[:2], img.shape[:2], "输出尺寸与输入不一致")
        ink = 1.0 - cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0 \
            if out.ndim == 3 else 1.0 - out.astype(np.float64) / 255.0
        # 房子黑色屋顶轮廓（原图 y=80..96 斜线区）应有结构响应
        roof = ink[80:100, 48:136]
        self.assertGreater(float(roof.mean()), 0.005, "像素图结构丢失")

    def test_scene_auto_changes_pipeline(self):
        """scene=auto 与 off 对拍照草图输出不同（自适应确实生效）。"""
        img = _load("sketch_photo.png")
        p_auto = LineArtParams()
        p_off = LineArtParams(scene="off")
        out_auto = to_lineart(img, p_auto)
        out_off = to_lineart(img, p_off)
        diff = np.abs(out_auto.astype(int) - out_off.astype(int)).mean()
        # SSR 渲染（v1.3.1+）使两条管线输出线宽共同收窄，均值差基线
        # 从 ~0.33 降到 ~0.29；阈值取 0.25 仍远高于噪声水平（<0.05）
        self.assertGreater(float(diff), 0.25, "auto 场景未改变输出")

    def test_apply_scene_params_override(self):
        """场景信息正确覆盖参数（sketch → adaptive 语义 + 局部自适应弱归一化）。"""
        img = _load("sketch_photo.png")
        _, p = _apply_scene(img, LineArtParams())
        self.assertEqual(p.semantic, "adaptive")
        self.assertEqual(p.illumination, 0.3)
        self.assertGreater(p.line_sensitivity, 0.6)

        img2 = _load("photo_like.png")
        _, p2 = _apply_scene(img2, LineArtParams())
        self.assertEqual(p2.semantic, "fdog")
        self.assertEqual(p2.coherence, 0.75)

    def test_explicit_scene(self):
        """显式指定场景也生效（不依赖检测）。"""
        img = _load("sketch_photo.png")
        _, p = _apply_scene(img, LineArtParams(scene="sketch"))
        self.assertEqual(p.semantic, "adaptive")

    def test_output_size_contract(self):
        """所有场景输出尺寸 = 输入尺寸。"""
        for name in ["sketch_photo.png", "blueprint.png", "pixel_art.png",
                     "document_scan.png", "photo_like.png"]:
            with self.subTest(sample=name):
                img = _load(name)
                out = to_lineart(img, LineArtParams())
                self.assertEqual(out.shape[:2], img.shape[:2])


if __name__ == "__main__":
    unittest.main()
