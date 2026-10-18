"""粗线双边缘合并（v1.7.0）：结构保真回归测试。

边缘检测的梯度脊线对线宽>=2px 的线给出两条（各在一侧），
v1.6.0 直接渲染把一根线画成两根伴随线（真线丢失 99%、虚假墨
1460px）。合并配对 + FWHM 线宽展开后应恢复"一根线画一根线"。
"""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import LineArtParams, to_lineart


def _make_scene():
    """256x256 白底 + 3px 硬边矩形 + 3px 水平线（灰度 40）。"""
    clean = np.full((256, 256), 255.0, np.float32)
    cv2.rectangle(clean, (60, 60), (196, 196), 40.0, 2)
    cv2.line(clean, (20, 128), (236, 128), 40.0, 2)
    return cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def _gt_mask():
    """真实墨位置（cv2 thickness=2 实测：边 59-61/195-197，线 127-129）。"""
    gt = np.zeros((256, 256), bool)
    gt[59:62, 59:198] = True
    gt[195:198, 59:198] = True
    gt[62:195, 59:62] = True
    gt[62:195, 195:198] = True
    gt[127:130, 20:237] = True
    return gt


class MergeEdgeTests(unittest.TestCase):
    def _convert(self, merge_edges, noise=False):
        bgr = _make_scene()
        if noise:
            rng = np.random.default_rng(7)
            blot = cv2.GaussianBlur(
                rng.normal(0, 12, (256, 256)).astype(np.float32), (31, 31), 10)
            grain = rng.normal(0, 8, (256, 256)).astype(np.float32)
            bgr = cv2.cvtColor(
                np.clip(bgr[:, :, 0].astype(np.float32) + blot * 10 + grain,
                        0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return to_lineart(bgr, LineArtParams(output="binary", mode="binary",
                                             merge_edges=merge_edges,
                                             scene="off"))

    def _metrics(self, out, gt):
        ink = out < 128
        fp = int(np.count_nonzero(ink & ~gt))
        miss = int(np.count_nonzero(~ink & gt))
        return fp, miss

    def test_double_edge_artifact_exists_without_merge(self):
        """v1.6.0 行为基线：不合并时双伴随线落在线外（结构性缺陷）。"""
        out = self._convert(merge_edges=False)
        gt = _gt_mask()
        ink = out < 128
        # 伴随线位置 [58,62] 与真实墨 [59,61] 不相交 → GT 内大面积丢失
        self.assertGreater(int(np.count_nonzero(~ink & gt)), 1500,
                           "基线丢失不足，测试图构造有误")

    def test_merge_restores_single_line(self):
        """合并后：单线心与真实墨位置吻合（干净图）。"""
        out = self._convert(merge_edges=True)
        y100 = np.where(out[100] < 128)[0].tolist()
        x128 = np.where(out[:, 128] < 128)[0].tolist()
        self.assertEqual(y100, [59, 60, 61, 195, 196, 197])
        self.assertEqual(x128, [59, 60, 61, 127, 128, 129,
                                195, 196, 197])

    def test_merge_precision(self):
        """合并精度：虚假墨 <40px、丢线 <5%（干净与噪声）。"""
        gt = _gt_mask()
        for noise in (False, True):
            with self.subTest(noise=noise):
                out = self._convert(merge_edges=True, noise=noise)
                fp, miss = self._metrics(out, gt)
                self.assertLess(fp, 40, f"虚假墨 {fp}px（噪声={noise}）")
                self.assertLess(miss, 0.05 * int(gt.sum()),
                                f"丢线 {miss}px（噪声={noise}）")

    def test_merge_gap_param(self):
        """merge_gap 参数透传：0 等价关闭合并。"""
        out = to_lineart(_make_scene(),
                         LineArtParams(output="binary", mode="binary",
                                       merge_edges=True, merge_gap=0,
                                       scene="off"))
        y100 = np.where(out[100] < 128)[0].tolist()
        # gap=0 找不到配对 → 双伴随线（同 merge=False）
        self.assertIn(58, y100)
        self.assertIn(62, y100)


if __name__ == "__main__":
    unittest.main()
