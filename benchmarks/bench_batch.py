"""批量处理性能基准测试：对比并行 vs 顺序处理。

测试场景：
1. 小批量（8 张）并行 vs 顺序加速比
2. 大批量（32 张）并行 vs 顺序加速比
3. 单张 baseline（验证并行无额外开销）
4. 错误处理：单张失败不阻塞批次
"""
import os
import sys
import time
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import LineArtParams, to_lineart, to_lineart_batch, to_lineart_batch_simple

# 合成测试图：192x192 白底黑圆
def make_test_circle(size=192, center=(96.0, 96.0), radius=40.0):
    img = np.full((size, size), 255.0, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = np.sqrt((yy - center[0]) ** 2 + (xx - center[1]) ** 2)
    coverage = np.clip(radius - d + 0.5, 0.0, 1.0)
    img = img - 200.0 * coverage
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


class BatchBenchmarkTests(unittest.TestCase):
    """批量 API 正确性与性能测试。"""

    def setUp(self):
        self.n_images = 8
        self.images = [make_test_circle() for _ in range(self.n_images)]
        self.params = LineArtParams()

    def test_batch_sequential_equivalence(self):
        """批量结果与逐个调用等价。"""
        results_single = [to_lineart(img, self.params) for img in self.images]
        results_batch = list(to_lineart_batch_simple(self.images, self.params, n_workers=1))
        for i, (single, batch) in enumerate(zip(results_single, results_batch)):
            np.testing.assert_array_equal(single, batch, f"第 {i} 张不一致")

    def test_batch_parallel_order(self):
        """并行输出顺序与输入一致。"""
        indices = []
        for idx, out, err in to_lineart_batch(self.images, self.params, n_workers=2):
            indices.append(idx)
        self.assertEqual(indices, list(range(len(self.images))))

    def test_batch_generator_input(self):
        """支持 generator 输入。"""
        def gen():
            for img in self.images:
                yield img

        results = list(to_lineart_batch_simple(gen(), self.params))
        self.assertEqual(len(results), len(self.images))

    def test_batch_empty_input(self):
        """空输入不报错。"""
        results = list(to_lineart_batch_simple([], self.params))
        self.assertEqual(results, [])

    def test_batch_error_handling(self):
        """单张失败不阻塞批次。"""
        bad_images = [self.images[0], None, self.images[2]]

        results = list(to_lineart_batch(bad_images, self.params, n_workers=1))
        self.assertEqual(len(results), 3)

        # 第一张成功
        self.assertEqual(results[0][0], 0)
        self.assertIsNone(results[0][2])
        self.assertIsNotNone(results[0][1])

        # 第二张失败
        self.assertEqual(results[1][0], 1)
        self.assertIsNotNone(results[1][2])
        self.assertIsNone(results[1][1])

        # 第三张成功
        self.assertEqual(results[2][0], 2)
        self.assertIsNone(results[2][2])
        self.assertIsNotNone(results[2][1])

    def test_batch_workers_default(self):
        """n_workers=None 自动检测 CPU 核数。"""
        from lineart.pipeline import _default_workers
        n = _default_workers()
        self.assertGreaterEqual(n, 1)

    def test_batch_simple_api(self):
        """简化版 API 只返回结果。"""
        results = list(to_lineart_batch_simple(self.images, self.params))
        self.assertEqual(len(results), len(self.images))
        for out in results:
            self.assertIsNotNone(out)

    def test_batch_progress_tqdm(self):
        """progress 参数不报错（tqdm 可选）。"""
        # 不安装 tqdm 时也能跑
        results = list(to_lineart_batch_simple(self.images, self.params, progress=True))
        self.assertEqual(len(results), len(self.images))

    def test_batch_n_workers_zero(self):
        """n_workers=0 退化为单线程。"""
        results = list(to_lineart_batch_simple(self.images, self.params, n_workers=0))
        self.assertEqual(len(results), len(self.images))


class BatchSpeedBenchmark(unittest.TestCase):
    """性能基准测试（仅输出加速比，不强制断言）。"""

    @unittest.skipUnless(os.getenv("RUN_BENCH") == "1", "设置 RUN_BENCH=1 运行性能基准")
    def test_parallel_speedup(self):
        """并行加速比：8 张图并行应快于顺序。"""
        images = [make_test_circle() for _ in range(8)]
        params = LineArtParams()

        # 顺序
        t0 = time.perf_counter()
        for img in images:
            to_lineart(img, params)
        t_seq = time.perf_counter() - t0

        # 并行
        t0 = time.perf_counter()
        list(to_lineart_batch_simple(images, params, n_workers=4))
        t_par = time.perf_counter() - t0

        speedup = t_seq / t_par
        print(f"\n===== 批量性能基准 =====")
        print(f"  顺序: {t_seq*1000:.1f} ms")
        print(f"  并行 (4 workers): {t_par*1000:.1f} ms")
        print(f"  加速比: {speedup:.2f}x")

    @unittest.skipUnless(os.getenv("RUN_BENCH") == "1", "设置 RUN_BENCH=1 运行性能基准")
    def test_large_batch(self):
        """大批量（32 张）并行性能。"""
        images = [make_test_circle() for _ in range(32)]
        params = LineArtParams()

        t0 = time.perf_counter()
        list(to_lineart_batch_simple(images, params, n_workers=4, progress=True))
        t_par = time.perf_counter() - t0
        print(f"\n  大批量 32 张: {t_par*1000:.1f} ms")
        print(f"  单张平均: {t_par/32*1000:.1f} ms")


if __name__ == "__main__":
    unittest.main(verbosity=2)
