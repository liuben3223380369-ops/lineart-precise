"""积分图 vs boxFilter 性能对比测试（guided_filter）。

测试内容：
1. 单引导滤波：积分图 vs boxFilter
2. 自引导快速路径：积分图 vs boxFilter
3. 不同图像尺寸下的性能对比
4. 数值精度验证
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart.preprocess.denoise import guided_filter, guided_filter_boxfilter


def bench(label, fn, n=5):
    """微基准测试：返回最小耗时（ms）。"""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(f"  {label:48s} {min(ts)*1000:8.1f} ms")
    return min(ts)


def make_test_image(h, w, noise=0.02, seed=42):
    """生成测试图像：渐变 + 噪声。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    img = 128.0 + 60.0 * np.sin(2 * np.pi * xx / 100) + 40.0 * yy / h
    img += rng.normal(0, noise * 255, (h, w))
    return np.clip(img, 0, 255).astype(np.float32)


def verify_numerical(a, b, tol=1e-4, name=""):
    """验证两个结果数值等价。"""
    diff = np.abs(a - b)
    max_diff = diff.max()
    mean_diff = diff.mean()
    status = "✓" if max_diff < tol else "✗"
    print(f"  {status} {name}: max|Δ|={max_diff:.2e}, mean|Δ|={mean_diff:.2e}")
    return max_diff < tol


def main():
    print("=" * 70)
    print("积分图 vs boxFilter 性能对比测试")
    print("=" * 70)

    # 测试配置
    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    radius = 3
    eps = 1e-4

    all_pass = True

    for h, w in sizes:
        print(f"\n{'─' * 70}")
        print(f"图像尺寸: {w}x{h} ({w*h/1e6:.2f} MP)")
        print(f"{'─' * 70}")

        img = make_test_image(h, w)
        guide = img  # 自引导

        # 1. 自引导快速路径对比
        print("\n[自引导快速路径]")
        def run_integral():
            return guided_filter(guide, img, radius=radius, eps=eps, use_integral=True)

        def run_boxfilter():
            return guided_filter_boxfilter(guide, img, radius=radius, eps=eps)

        r_integral = run_integral()
        r_box = run_boxfilter()

        all_pass &= verify_numerical(r_integral, r_box, 1e-4, "自引导数值等价")

        bench("积分图（自引导）", run_integral)
        bench("boxFilter（自引导）", run_boxfilter)

        # 2. 非自引导对比
        print("\n[非自引导]")
        guide2 = img * 0.8 + 0.1  # 不同的引导图像

        def run_integral_noguide():
            return guided_filter(guide2, img, radius=radius, eps=eps, use_integral=True)

        def run_boxfilter_noguide():
            return guided_filter_boxfilter(guide2, img, radius=radius, eps=eps)

        r_integral2 = run_integral_noguide()
        r_box2 = run_boxfilter_noguide()

        all_pass &= verify_numerical(r_integral2, r_box2, 1e-4, "非自引导数值等价")

        bench("积分图（非自引导）", run_integral_noguide)
        bench("boxFilter（非自引导）", run_boxfilter_noguide)

    # 3. 不同半径对比
    print(f"\n{'─' * 70}")
    print("不同半径性能对比（1024x1024）")
    print(f"{'─' * 70}")

    img = make_test_image(1024, 1024)
    for r in [2, 4, 8, 16]:
        print(f"\n半径 r={r}:")
        def run_r_int():
            return guided_filter(img, img, radius=r, eps=eps, use_integral=True)
        def run_r_box():
            return guided_filter_boxfilter(img, img, radius=r, eps=eps)

        r1 = run_r_int()
        r2 = run_r_box()
        all_pass &= verify_numerical(r1, r2, 1e-4, f"r={r} 等价")
        bench(f"  积分图 r={r}", run_r_int)
        bench(f"  boxFilter r={r}", run_r_box)

    # 4. 精度验证（小图详细对比）
    print(f"\n{'─' * 70}")
    print("数值精度详细验证（64x64）")
    print(f"{'─' * 70}")

    small = make_test_image(64, 64, seed=99)
    r_int = guided_filter(small, small, radius=3, eps=1e-4, use_integral=True)
    r_box = guided_filter_boxfilter(small, small, radius=3, eps=1e-4)

    diff = np.abs(r_int - r_box)
    print(f"  最大绝对误差: {diff.max():.2e}")
    print(f"  平均绝对误差: {diff.mean():.2e}")
    print(f"  误差 > 1e-6 像素数: {(diff > 1e-6).sum()}")
    print(f"  误差 > 1e-5 像素数: {(diff > 1e-5).sum()}")

    # 5. 边界处理验证
    print(f"\n{'─' * 70}")
    print("边界处理验证（BORDER_REPLICATE）")
    print(f"{'─' * 70}")

    # 创建边缘明显的图像
    edge_img = np.ones((64, 64), dtype=np.float32) * 0.5
    edge_img[:, :16] = 0.2  # 左侧深色
    edge_img[:, -16:] = 0.8  # 右侧浅色

    r_edge_int = guided_filter(edge_img, edge_img, radius=4, eps=1e-4, use_integral=True)
    r_edge_box = guided_filter_boxfilter(edge_img, edge_img, radius=4, eps=1e-4)

    # 检查边界区域
    border_diff = np.abs(r_edge_int - r_edge_box)
    print(f"  边界区域（前4行）max|Δ|: {border_diff[:4].max():.2e}")
    print(f"  边界区域（后4行）max|Δ|: {border_diff[-4:].max():.2e}")
    print(f"  边界区域（前4列）max|Δ|: {border_diff[:, :4].max():.2e}")
    print(f"  边界区域（后4列）max|Δ|: {border_diff[:, -4:].max():.2e}")

    # 总结
    print(f"\n{'=' * 70}")
    if all_pass:
        print("✓ 所有数值等价性测试通过")
    else:
        print("✗ 部分数值等价性测试失败")
    print(f"{'=' * 70}")

    return all_pass


if __name__ == "__main__":
    main()
