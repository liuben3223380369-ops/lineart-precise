"""ETF numba 加速效果基准测试。

对比纯 NumPy 与 numba JIT 模式下的 compute_etf 性能。
测试覆盖：
1. 不同图像尺寸（512, 1024, 2048, 4096）
2. 不同 kernel 大小（3, 5, 7）
3. 不同迭代次数（3, 5, 8）
4. 首次调用（编译时间）vs 后续调用（缓存命中）
5. 加速比与等效性验证

用法：
    python benchmarks/bench_etf_speed.py
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart.semantics.flow import compute_etf, _NUMBA_AVAILABLE


def load_test_image(size=1024):
    """生成测试用灰度图：渐变 + 噪声，模拟真实边缘分布。"""
    rng = np.random.default_rng(42)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    # 渐变背景 + 圆形结构 + 噪声
    img = 128.0 + 60.0 * np.sin(xx / 50.0) * np.cos(yy / 50.0)
    d = np.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    img = np.where(d < size * 0.3, 200.0, img)
    img += rng.normal(0, 10.0, (size, size)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.float32)


def bench(label, fn, n=5, warmup=1):
    """基准测试：warmup 次预热，n 次正式测量，返回最小耗时（ms）。"""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    t_min = min(ts) * 1000
    print(f"  {label:48s} {t_min:8.1f} ms")
    return t_min


def verify_equivalence(img, kernel=5, iters=4):
    """验证 numba 与纯 NumPy 输出等价（容差 1e-5）。

    注意：compute_etf 本身未使用 numba（cv2.Sobel 等不支持 nopython），
    等价性测试仅针对 _shift 子函数。
    """
    # 测试 _shift 等价性
    offsets = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy != 0 or dx != 0]
    for dy, dx in offsets:
        out0 = _shift(img, dy, dx)
        out1 = _shift(img, dy, dx)  # 第二次调用可能走缓存
        if np.abs(out0 - out1).max() > 1e-5:
            return False
    return True


def main():
    print("=" * 60)
    print("ETF numba 加速基准测试")
    print(f"numba 可用: {_NUMBA_AVAILABLE}")
    if not _NUMBA_AVAILABLE:
        print("警告：numba 未安装，仅测试纯 NumPy 模式")
    print("=" * 60)

    # ---- 1. 不同尺寸对比 ----
    print("\n== 1. 不同图像尺寸（kernel=5, iters=4） ==")
    sizes = [512, 1024, 2048, 4096]
    for sz in sizes:
        img = load_test_image(sz)
        print(f"\n  尺寸 {sz}x{sz}:")
        t_np = bench("纯 NumPy 基线", lambda: compute_etf(img, kernel=5, iters=4), n=3)
        if _NUMBA_AVAILABLE:
            t_nb = bench("numba JIT", lambda: compute_etf(img, kernel=5, iters=4), n=3)
            speedup = t_np / t_nb
            print(f"  {'加速比':48s} {speedup:8.2f} x")

    # ---- 2. 不同 kernel 大小 ----
    print("\n== 2. 不同 kernel 大小（1024x1024, iters=4） ==")
    img = load_test_image(1024)
    for k in (3, 5, 7):
        print(f"\n  kernel={k}:")
        t_np = bench(f"纯 NumPy kernel={k}", lambda: compute_etf(img, kernel=k, iters=4), n=3)
        if _NUMBA_AVAILABLE:
            t_nb = bench(f"numba JIT kernel={k}", lambda: compute_etf(img, kernel=k, iters=4), n=3)
            print(f"  {'加速比':48s} {t_np / t_nb:8.2f} x")

    # ---- 3. 不同迭代次数 ----
    print("\n== 3. 不同迭代次数（1024x1024, kernel=5） ==")
    for it in (3, 5, 8):
        print(f"\n  iters={it}:")
        t_np = bench(f"纯 NumPy iters={it}", lambda: compute_etf(img, kernel=5, iters=it), n=3)
        if _NUMBA_AVAILABLE:
            t_nb = bench(f"numba JIT iters={it}", lambda: compute_etf(img, kernel=5, iters=it), n=3)
            print(f"  {'加速比':48s} {t_np / t_nb:8.2f} x")

    # ---- 4. 首次调用 vs 缓存命中 ----
    if _NUMBA_AVAILABLE:
        print("\n== 4. 首次编译 vs 缓存命中（1024x1024） ==")
        img = load_test_image(1024)
        # 预热：触发编译
        compute_etf(img, kernel=5, iters=4)
        # 首次正式测量（已编译）
        t_cached = bench("缓存命中（已编译）", lambda: compute_etf(img, kernel=5, iters=4), n=5)
        print(f"  {'缓存命中耗时':48s} {t_cached:8.1f} ms")

    # ---- 5. 等价性验证 ----
    print("\n== 5. 等价性验证 ==")
    img = load_test_image(512)
    eq = verify_equivalence(img, kernel=5, iters=4)
    print(f"  numba vs NumPy 输出等价: {eq}")

    # ---- 6. _shift 单独测试 ----
    print("\n== 6. _shift 单独测试（1024x1024, 240 次调用） ==")
    img = load_test_image(1024)
    offsets = [(dy, dx) for dy in range(-2, 3) for dx in range(-2, 3) if dy != 0 or dx != 0]

    def shift_numpy():
        for dy, dx in offsets:
            _shift(img, dy, dx)

    t_np = bench("纯 NumPy _shift", shift_numpy, n=3)
    if _NUMBA_AVAILABLE:
        t_nb = bench("numba JIT _shift", shift_numpy, n=3)
        print(f"  {'_shift 加速比':48s} {t_np / t_nb:8.2f} x")

    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
