"""D1/D2/D3 新场景检测基准测试。

测试场景：
- D1 漫画场景：高对比度黑白色块 + 网点纹理
- D2 手写笔记场景：潦草字迹 + 弱线条 + 不均匀光照
- D3 建筑图纸场景：规则网格 + 细线 + 高对比

用法:
  python3 benchmarks/bench_new_scenes.py          # 运行所有测试
  python3 benchmarks/bench_new_scenes.py --report  # 输出详细报告
"""
import os
import sys
import time
import unittest
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import LineArtParams, to_lineart
from lineart.scenes import (
    detect_scene,
    scene_info_for,
    detect_comic_scene,
    detect_handwriting_scene,
    detect_architectural_scene,
)

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "demo", "samples")


def _load(name):
    return cv2.imread(os.path.join(SAMPLES, name))


def _synthesize_comic():
    """合成漫画测试图：黑白色块 + 网点纹理。"""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    # 黑色块
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 0), -1)
    # 白色块
    cv2.rectangle(img, (130, 20), (210, 100), (255, 255, 255), -1)
    # 灰色块
    cv2.rectangle(img, (20, 130), (100, 210), (180, 180, 180), -1)
    # 网点纹理区域
    for y in range(130, 210, 4):
        for x in range(130, 210, 4):
            if (x + y) % 8 < 4:
                img[y:y+2, x:x+2] = (150, 150, 150)
    # 黑色粗线
    cv2.line(img, (20, 120), (210, 120), (0, 0, 0), 3)
    return img


def _synthesize_handwriting():
    """合成手写笔记测试图：潦草字迹 + 弱线条 + 不均匀光照。"""
    # 不均匀光照背景
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(256):
        for c in range(3):
            img[:, :, c] = 200 + 40 * np.sin(y / 40 * np.pi)
    # 弱线条（潦草字迹）
    rng = np.random.default_rng(42)
    for i in range(15):
        x = rng.integers(20, 230)
        y = rng.integers(20, 230)
        pts = np.column_stack([
            x + rng.integers(-15, 16, 8),
            y + rng.integers(-10, 11, 8)
        ]).astype(np.int32)
        cv2.polylines(img, [pts], False, (80, 80, 80), 2, cv2.LINE_AA)
    # 添加一些噪点
    noise = rng.normal(0, 5, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


def _synthesize_architectural():
    """合建筑图纸测试图：规则网格 + 细线 + 高对比。"""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    # 规则网格
    for i in range(0, 256, 16):
        cv2.line(img, (i, 0), (i, 255), (200, 200, 200), 1)
        cv2.line(img, (0, i), (256, i), (200, 200, 200), 1)
    # 粗线结构
    cv2.rectangle(img, (50, 50), (200, 200), (50, 50, 50), 2)
    # 内部细节
    for i in range(60, 190, 20):
        cv2.line(img, (50, i), (200, i), (100, 100, 100), 1)
    # 标注文字
    cv2.putText(img, "A-1", (100, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img


class ComicSceneTests(unittest.TestCase):
    """D1 漫画场景检测测试。"""

    def test_detect_synthesized_comic(self):
        """合成漫画图应被检测为 comic 场景。"""
        img = _synthesize_comic()
        info = detect_scene(img)
        self.assertEqual(info.scene, "comic",
                         f"检测为 {info.scene} 而非 comic")
        self.assertGreaterEqual(info.confidence, 0.35,
                                f"置信度过低: {info.confidence}")

    def test_comic_params_override(self):
        """comic 场景应覆盖为 adaptive 语义 + 高 sensitivity。"""
        img = _synthesize_comic()
        info = detect_scene(img)
        self.assertEqual(info.params.get("semantic"), "adaptive")
        self.assertGreaterEqual(info.params.get("line_sensitivity", 0), 0.8,
                                f"line_sensitivity 不够高: {info.params.get('line_sensitivity')}")

    def test_detect_function_direct(self):
        """直接调用 detect_comic_scene 函数。"""
        img = _synthesize_comic()
        conf = detect_comic_scene(img)
        self.assertGreater(conf, 0.35,
                           f"直接检测置信度过低: {conf}")

    def test_comic_conversion_quality(self):
        """漫画图转换应保留线条。"""
        img = _synthesize_comic()
        out = to_lineart(img, LineArtParams(scene="comic"))
        ink = 1.0 - out.astype(np.float64) / 255.0
        # 黑色块区域应有墨迹
        black_ink = ink[20:100, 20:100]
        self.assertGreater(float(black_ink.mean()), 0.5,
                           "黑色块区域墨迹丢失")


class HandwritingSceneTests(unittest.TestCase):
    """D2 手写笔记场景检测测试。"""

    def test_detect_synthesized_handwriting(self):
        """合成手写笔记图应被检测为 handwriting 场景。"""
        img = _synthesize_handwriting()
        info = detect_scene(img)
        self.assertEqual(info.scene, "handwriting",
                         f"检测为 {info.scene} 而非 handwriting")
        self.assertGreaterEqual(info.confidence, 0.35,
                                f"置信度过低: {info.confidence}")

    def test_handwriting_params_override(self):
        """handwriting 场景应覆盖为 fdog 语义 + 高 detail + 高 coherence。"""
        img = _synthesize_handwriting()
        info = detect_scene(img)
        self.assertEqual(info.params.get("semantic"), "fdog")
        self.assertGreaterEqual(info.params.get("detail", 0), 0.8,
                                f"detail 不够高: {info.params.get('detail')}")
        self.assertGreaterEqual(info.params.get("coherence", 0), 0.8,
                                f"coherence 不够高: {info.params.get('coherence')}")

    def test_detect_function_direct(self):
        """直接调用 detect_handwriting_scene 函数。"""
        img = _synthesize_handwriting()
        conf = detect_handwriting_scene(img)
        self.assertGreater(conf, 0.30,
                           f"直接检测置信度过低: {conf}")

    def test_handwriting_conversion_quality(self):
        """手写笔记图转换应保留弱线条。"""
        img = _synthesize_handwriting()
        out = to_lineart(img, LineArtParams(scene="handwriting"))
        ink = 1.0 - out.astype(np.float64) / 255.0
        # 应有墨迹响应
        self.assertGreater(float(ink.mean()), 0.01,
                           "线条墨迹丢失")


class ArchitecturalSceneTests(unittest.TestCase):
    """D3 建筑图纸场景检测测试。"""

    def test_detect_synthesized_architectural(self):
        """合成建筑图纸图应被检测为 architectural 场景。"""
        img = _synthesize_architectural()
        info = detect_scene(img)
        self.assertEqual(info.scene, "architectural",
                         f"检测为 {info.scene} 而非 architectural")
        self.assertGreaterEqual(info.confidence, 0.35,
                                f"置信度过低: {info.confidence}")

    def test_architectural_params_override(self):
        """architectural 场景应覆盖 remove_grid=True + 高 sensitivity。"""
        img = _synthesize_architectural()
        info = detect_scene(img)
        self.assertTrue(info.params.get("remove_grid", False),
                        f"remove_grid 未启用: {info.params.get('remove_grid')}")
        self.assertGreaterEqual(info.params.get("line_sensitivity", 0), 0.7,
                                f"line_sensitivity 不够高: {info.params.get('line_sensitivity')}")

    def test_detect_function_direct(self):
        """直接调用 detect_architectural_scene 函数。"""
        img = _synthesize_architectural()
        conf = detect_architectural_scene(img)
        self.assertGreater(conf, 0.35,
                           f"直接检测置信度过低: {conf}")

    def test_architectural_conversion_quality(self):
        """建筑图纸图转换应保留细线。"""
        img = _synthesize_architectural()
        out = to_lineart(img, LineArtParams(scene="architectural"))
        ink = 1.0 - out.astype(np.float64) / 255.0
        # 网格线应有响应
        grid_ink = ink[::16, ::16]
        self.assertGreater(float(grid_ink.mean()), 0.05,
                           "网格线墨迹丢失")


class ScenePriorityTests(unittest.TestCase):
    """场景检测优先级测试。"""

    def test_architectural_before_sketch(self):
        """建筑图纸应优先于 sketch/lineart 检测。"""
        img = _synthesize_architectural()
        info = detect_scene(img)
        self.assertEqual(info.scene, "architectural",
                         f"建筑图纸检测为 {info.scene}，优先级错误")

    def test_comic_before_lineart(self):
        """漫画应优先于 lineart 检测。"""
        img = _synthesize_comic()
        info = detect_scene(img)
        self.assertEqual(info.scene, "comic",
                         f"漫画检测为 {info.scene}，优先级错误")

    def test_handwriting_before_photo(self):
        """手写笔记应优先于 photo 检测。"""
        img = _synthesize_handwriting()
        info = detect_scene(img)
        self.assertEqual(info.scene, "handwriting",
                         f"手写笔记检测为 {info.scene}，优先级错误")


class BackwardCompatibilityTests(unittest.TestCase):
    """向后兼容性测试。"""

    def test_explicit_scene_comic(self):
        """显式指定 comic 场景应生效。"""
        img = _synthesize_comic()
        info = scene_info_for("comic")
        self.assertEqual(info.scene, "comic")
        self.assertEqual(info.confidence, 1.0)

    def test_explicit_scene_handwriting(self):
        """显式指定 handwriting 场景应生效。"""
        img = _synthesize_handwriting()
        info = scene_info_for("handwriting")
        self.assertEqual(info.scene, "handwriting")
        self.assertEqual(info.confidence, 1.0)

    def test_explicit_scene_architectural(self):
        """显式指定 architectural 场景应生效。"""
        img = _synthesize_architectural()
        info = scene_info_for("architectural")
        self.assertEqual(info.scene, "architectural")
        self.assertEqual(info.confidence, 1.0)

    def test_unknown_scene_falls_back(self):
        """未知场景应回退到 lineart。"""
        info = scene_info_for("unknown_scene")
        self.assertEqual(info.scene, "lineart")

    def test_existing_scenes_still_work(self):
        """原有场景仍应正常检测。"""
        # 蓝图
        img = _load("blueprint.png")
        if img is not None:
            info = detect_scene(img)
            self.assertEqual(info.scene, "blueprint")
        # 像素艺术
        img = _load("pixel_art.png")
        if img is not None:
            info = detect_scene(img)
            self.assertEqual(info.scene, "pixel")
        # 拍照手绘
        img = _load("sketch_photo.png")
        if img is not None:
            info = detect_scene(img)
            self.assertEqual(info.scene, "sketch")


class PerformanceTests(unittest.TestCase):
    """性能测试。"""

    def test_comic_detection_speed(self):
        """漫画场景检测应在合理时间内完成。"""
        img = _synthesize_comic()
        start = time.perf_counter()
        for _ in range(10):
            detect_comic_scene(img)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0,
                        f"检测耗时过长: {elapsed:.3f}s")

    def test_handwriting_detection_speed(self):
        """手写笔记场景检测应在合理时间内完成。"""
        img = _synthesize_handwriting()
        start = time.perf_counter()
        for _ in range(10):
            detect_handwriting_scene(img)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0,
                        f"检测耗时过长: {elapsed:.3f}s")

    def test_architectural_detection_speed(self):
        """建筑图纸场景检测应在合理时间内完成。"""
        img = _synthesize_architectural()
        start = time.perf_counter()
        for _ in range(10):
            detect_architectural_scene(img)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0,
                        f"检测耗时过长: {elapsed:.3f}s")

    def test_full_detection_speed(self):
        """完整场景检测应在合理时间内完成。"""
        img = _synthesize_comic()
        start = time.perf_counter()
        for _ in range(10):
            detect_scene(img)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0,
                        f"完整检测耗时过长: {elapsed:.3f}s")


if __name__ == "__main__":
    unittest.main(verbosity=2)
