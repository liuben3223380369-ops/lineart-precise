"""场景误判与参数覆盖语义（v1.13.0）回归测试。

背景（用户实测反馈三类问题）：
1. 黑底白线 CAD 图被误判为 pixel 场景（颜色少 + 硬边与像素艺术
   无法区分）→ 4x 预放大 + thickness=4 + merge_edges=False 导致
   双线化/文字重影/实心点空心化，笔画保留率仅 5%。
   修复：_blocky_solid_ratio 块状实心率判据（细线 <=0.18 vs 块状 0.97）。
2. 场景默认参数静默覆盖用户显式参数（照片调 denoise 被 photo 场景
   0.55 顶掉，无法调优）。修复：场景值只覆盖仍等于 dataclass 默认的字段。
3. normalize_illumination 收到 uint8 输入时 strength<1 分支会把整图
   推成全白。修复：入口归一化到 [0,1] float。
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart.params import LineArtParams
from lineart.pipeline import _apply_scene, to_lineart
from lineart.preprocess.illumination import normalize_illumination
from lineart.scenes import _blocky_solid_ratio, detect_scene

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "demo", "samples")


def _make_cad_black_bg(w=640, h=480):
    """黑底白线 CAD 图：矩形 + 圆 + 中心线 + 刻度 + 文字（用户场景）。"""
    img = np.zeros((h, w, 3), np.uint8)
    cv2.rectangle(img, (80, 80), (560, 400), (255, 255, 255), 2)
    cv2.circle(img, (320, 240), 60, (255, 255, 255), 2)
    cv2.line(img, (250, 240), (390, 240), (255, 255, 255), 1)
    cv2.line(img, (320, 170), (320, 310), (255, 255, 255), 1)
    cv2.putText(img, "SECTION A-A", (80, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


class TestBlockySolidRatio(unittest.TestCase):
    """块状实心率判据的区分度。"""

    def test_pixel_art_high(self):
        img = cv2.imread(os.path.join(SAMPLES, "pixel_art.png"))
        self.assertGreater(_blocky_solid_ratio(img), 0.5)

    def test_thin_line_cad_low(self):
        self.assertLess(_blocky_solid_ratio(_make_cad_black_bg()), 0.5)

    def test_blueprint_low(self):
        img = cv2.imread(os.path.join(SAMPLES, "blueprint.png"))
        self.assertLess(_blocky_solid_ratio(img), 0.5)

    def test_solid_blocks_high(self):
        """合成大色块图（像素艺术特征）应判高。"""
        img = np.zeros((96, 96, 3), np.uint8)
        img[10:50, 10:50] = (200, 40, 40)
        img[50:90, 50:90] = (40, 40, 200)
        self.assertGreater(_blocky_solid_ratio(img), 0.5)


class TestCadBlackSceneRouting(unittest.TestCase):
    """黑底白线 CAD 图不再误判为 pixel。"""

    def test_cad_black_not_pixel(self):
        info = detect_scene(_make_cad_black_bg())
        self.assertNotEqual(info.scene, "pixel")
        self.assertEqual(info.scene, "lineart")

    def test_official_samples_unchanged(self):
        """官方样本场景判定不回归。"""
        expected = {
            "pixel_art.png": "pixel",
            "blueprint.png": "blueprint",
            "cad_black_color.png": "lineart",
            "sketch_photo.png": "sketch",
            "photo_like.png": "photo",
        }
        for name, want in expected.items():
            with self.subTest(sample=name):
                img = cv2.imread(os.path.join(SAMPLES, name))
                self.assertEqual(detect_scene(img).scene, want)

    def test_cad_black_conversion_retains_strokes(self):
        """端到端：笔画保留率应远高于误判时的 5%。"""
        img = _make_cad_black_bg()
        out = to_lineart(img, LineArtParams())
        src_ink = int((cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 128).sum())
        ink = int((out < 128).sum())
        self.assertGreater(ink / src_ink, 0.5,
                           f"笔画保留率过低: {ink / src_ink:.1%}")


class TestSceneParamOverrideSemantics(unittest.TestCase):
    """v1.13.0 场景参数覆盖语义：用户显式参数优先。"""

    IMG = _make_cad_black_bg()

    def test_user_param_wins(self):
        _, p = _apply_scene(self.IMG, LineArtParams(scene="photo", denoise=0.1))
        self.assertEqual(p.denoise, 0.1)

    def test_untouched_param_gets_scene_value(self):
        _, p = _apply_scene(self.IMG, LineArtParams(scene="photo"))
        self.assertEqual(p.denoise, 0.55)
        self.assertEqual(p.coherence, 0.75)

    def test_mixed(self):
        """用户改一半：改的保持、没改的用场景值。

        注：coherence 取 0.65（既非默认 0.6 也非 photo 场景 0.75），
        避开「显式传默认值无法与未传区分」的语义边界。
        """
        _, p = _apply_scene(self.IMG, LineArtParams(
            scene="photo", denoise=0.1, coherence=0.65))
        self.assertEqual(p.denoise, 0.1)
        self.assertEqual(p.coherence, 0.65)
        self.assertEqual(p.gradient_gamma, 0.6)  # photo 场景值

    def test_default_valued_user_param_still_overridden(self):
        """用户显式传默认值与不传无法区分，按场景覆盖（语义边界）。"""
        _, p = _apply_scene(self.IMG, LineArtParams(scene="photo", denoise=0.4))
        self.assertEqual(p.denoise, 0.55)

    def test_scene_off_no_override(self):
        _, p = _apply_scene(self.IMG, LineArtParams(scene="off", denoise=0.1))
        self.assertEqual(p.denoise, 0.1)


class TestIlluminationUint8Guard(unittest.TestCase):
    """normalize_illumination 的 uint8 输入防御。"""

    def test_uint8_input_not_all_white(self):
        gray = np.full((64, 64), 128, np.uint8)
        gray[20:44, 20:44] = 30
        out = normalize_illumination(gray, strength=0.8)
        self.assertEqual(out.dtype, np.float32)
        self.assertLess(float(out.max()), 1.0 + 1e-6)
        self.assertFalse(bool((out > 0.99).all()), "uint8 输入不应整图全白")

    def test_float_input_unchanged_behavior(self):
        gray = np.full((64, 64), 0.5, np.float32)
        gray[20:44, 20:44] = 0.1
        out = normalize_illumination(gray, strength=0.8)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertGreater(float(out.std()), 0.0)


if __name__ == "__main__":
    unittest.main()
