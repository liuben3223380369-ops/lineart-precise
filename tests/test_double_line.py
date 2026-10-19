"""双线化回归测试（v1.15.0）。

背景（用户实测反馈）：白底黑粗线图输出双线——一条线被画成两条。
三层根因与修复：
1. 场景误判：粗线图被 detect_scene 判为 pixel（场景参数
   merge_edges=False + thickness=4 导致双边缘脊线不合并）。
   旧判据 _blocky_solid_ratio（DT>1.5 全局占比）对粗线失效
   （12px 线内部同样大片距边缘 >1.5px，实心率 0.67 > 阈值 0.5）。
   修复：_deep_core_ratio 深核判据——深核 = DT>=12 区域，对深核
   掩码做连通域分析（细长度>=8 排除超粗线芯、填充率<0.3 排除
   环形芯带），只有大实心色块才有深核（pixel_art 0.636），
   线状结构深核只是中心细条带（各宽度线/环/框均 0.000），
   阈值 0.15。
2. merge_gap 上限 6 只覆盖 ~7px 线宽，8px+ 粗线双边缘配对距离
   超限退化为双线。修复：默认 6->20（params.py 默认值 +
   render.py / pipeline.py 函数签名三处同步）。
3. 暗谷判据太松（gap 扩大的副作用）：dark_ratio 0.75 时盘外
   灰背景恰好通过宽松判据，噪声点远距误配对把中点渲染进圆盘
   （test_circle_noisy 曾失败）。修复：0.75->0.6。

断言方向设计（双线 vs 断裂是两种不同的失败模式）：
- 双线检测：垂直于线方向扫描（水平线列扫、垂直线行扫），
  单线 = 1 段，双线 = 2 段。
- 断裂检测：沿线方向扫描中段（避开端点/拐角 1px 毛刺，
  v1.14.0 既有渲染层边界行为），连续 = 1 段。
- 粗线渲染为 FWHM 宽度条带（线宽保真），非 1px 细线，
  故垂直扫描的黑段本身有一定宽度，断言只看段数。
- 圆取过圆心列扫描：上弧 + 下弧各一段 = 2 段（双线圆为 4 段）。
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart.pipeline import to_lineart
from lineart.scenes import detect_scene

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "demo", "samples")


def _segments(mask):
    """01 掩码的连续黑段数。"""
    return int(np.sum(mask[1:] & ~mask[:-1]) + mask[0])


class TestSceneRouting(unittest.TestCase):
    """第一层修复：粗线图不得误判 pixel（pixel 场景参数导致双线）。"""

    def test_thick_lines_not_pixel(self):
        for th in (8, 12, 16, 20):
            img = np.full((240, 240), 255, np.uint8)
            cv2.line(img, (40, 120), (200, 120), 0, th)
            self.assertEqual(
                detect_scene(img).scene, "lineart",
                f"{th}px 水平粗线被误判，场景参数会导致双线")

    def test_thick_ring_not_pixel(self):
        img = np.full((240, 240), 255, np.uint8)
        cv2.circle(img, (120, 120), 60, 0, 20)
        self.assertEqual(detect_scene(img).scene, "lineart")

    def test_thick_frame_not_pixel(self):
        img = np.full((240, 240), 255, np.uint8)
        cv2.rectangle(img, (70, 70), (170, 170), 0, 16)
        self.assertEqual(detect_scene(img).scene, "lineart")

    def test_pixel_art_still_pixel(self):
        """防误伤：官方像素艺术样本仍判 pixel（深核占比高）。"""
        img = cv2.imread(os.path.join(SAMPLES, "pixel_art.png"))
        self.assertEqual(detect_scene(img).scene, "pixel")


class TestStraightLines(unittest.TestCase):
    """第二层修复：各宽度直线单线。

    双线检测 = 垂直于线扫描 1 段；断裂检测 = 沿线扫描中段 1 段
    （中段避开端点毛刺）。
    """

    def test_horizontal_single_stroke(self):
        for th in (2, 4, 6, 8, 12, 16, 20):
            img = np.full((220, 220), 255, np.uint8)
            cv2.line(img, (40, 110), (180, 110), 0, th)
            out = to_lineart(img)
            ys = np.where(np.any(out[:, 90:130] < 128, axis=1))[0]
            self.assertGreater(len(ys), 0, f"{th}px 水平线未检出")
            for x in range(85, 136, 5):
                n = _segments(out[:, x] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 水平线列扫 x={x} 得 {n} 段（双线/空洞）")
            for yc in ys:
                n = _segments(out[yc, 85:135] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 水平线 y={yc} 中段断裂（{n} 段）")

    def test_vertical_single_stroke(self):
        for th in (2, 4, 6, 8, 12, 16, 20):
            img = np.full((220, 220), 255, np.uint8)
            cv2.line(img, (110, 40), (110, 180), 0, th)
            out = to_lineart(img)
            xs = np.where(np.any(out[90:130, :] < 128, axis=0))[0]
            self.assertGreater(len(xs), 0, f"{th}px 垂直线未检出")
            for y in range(85, 136, 5):
                n = _segments(out[y, :] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 垂直线行扫 y={y} 得 {n} 段（双线/空洞）")
            for xc in xs:
                n = _segments(out[85:135, xc] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 垂直线 x={xc} 中段断裂（{n} 段）")

    def test_gray_thick_line_single_stroke(self):
        """第三层修复验证：灰色粗线（暗谷边界）不因 dark_ratio 收紧丢线。"""
        for gray in (100, 150, 180):
            img = np.full((220, 220), 255, np.uint8)
            cv2.line(img, (40, 110), (180, 110), gray, 12)
            out = to_lineart(img)
            ys = np.where(np.any(out[:, 90:130] < 128, axis=1))[0]
            self.assertGreater(len(ys), 0, f"灰{gray} 粗线未检出")
            for x in range(60, 161, 10):
                n = _segments(out[:, x] < 128)
                self.assertEqual(
                    n, 1,
                    f"灰{gray} 12px 线列扫 x={x} 得 {n} 段（双线/空洞）")


class TestCircles(unittest.TestCase):
    """各宽度圆：过圆心列扫描 = 上弧+下弧 2 段（双线圆为 4 段）。"""

    def test_circle_two_arcs(self):
        for th in (3, 6, 10, 16, 20):
            img = np.full((240, 240), 255, np.uint8)
            cv2.circle(img, (120, 120), 60, 0, th)
            out = to_lineart(img)
            n = _segments(out[35:205, 120] < 128)
            self.assertEqual(n, 2, f"{th}px 圆过圆心列扫得 {n} 段（期望 2）")

    def test_noisy_thick_circle_two_arcs(self):
        """噪声鲁棒：粗线圆加噪声后不得误配对出多余段。"""
        rng = np.random.default_rng(42)
        img = np.full((240, 240), 255, np.uint8)
        cv2.circle(img, (120, 120), 60, 0, 10)
        noisy = np.clip(
            img.astype(np.float64) + rng.normal(0, 25, img.shape),
            0, 255).astype(np.uint8)
        out = to_lineart(noisy)
        n = _segments(out[35:205, 120] < 128)
        self.assertEqual(n, 2, f"噪声 10px 圆得 {n} 段（期望 2）")


class TestRectFrames(unittest.TestCase):
    """粗空心框：四条边各自单线，中线扫描见左右框线 2 段。

    框边扫描取中段范围（x/y 60-90 内的边区域），避开拐角毛刺
    （v1.14.0 既有边界行为）。
    """

    def test_frame_edges_single_stroke(self):
        for th in (6, 10, 16):
            img = np.full((240, 240), 255, np.uint8)
            cv2.rectangle(img, (70, 70), (170, 170), 0, th)
            out = to_lineart(img)

            # 中线行扫：左右框线各 1 段（双线则 4 段）
            for y in range(105, 140, 7):
                n = _segments(out[y, 60:181] < 128)
                self.assertEqual(n, 2, f"{th}px 框 y={y} 行扫得 {n} 段")

            # 左框线（垂直线）双线检测：行扫边区域中段各行 1 段
            for y in range(100, 146, 5):
                n = _segments(out[y, 60:90] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 框左线 y={y} 行扫得 {n} 段（双线）")

            # 上框线（水平线）双线检测：列扫边区域中段各列 1 段
            for x in range(100, 146, 5):
                n = _segments(out[60:90, x] < 128)
                self.assertEqual(
                    n, 1, f"{th}px 框上线 x={x} 列扫得 {n} 段（双线）")


if __name__ == "__main__":
    unittest.main()
