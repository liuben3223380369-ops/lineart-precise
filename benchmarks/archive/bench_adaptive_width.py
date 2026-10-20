"""自适应线宽估计基准测试：验证硬边线/软线的 FWHM 阈值自适应。

硬边线（高梯度幅值）：阈值 c+0.4(bg-c)（更窄）
软线（低梯度幅值）：阈值 c+0.6(bg-c)（更宽）
分界点：0.5*max_gradient

用法:
    python3 benchmarks/bench_adaptive_width.py
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
import time
import cv2
import numpy as np

from lineart import to_lineart, LineArtParams


def make_hard_edge_line(size=256):
    """硬边线：高对比度阶跃边缘，梯度幅值高。"""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (100, 80), (150, 180), (0, 0, 0), -1)
    return img


def make_soft_line(size=256):
    """软线：低对比度渐变边缘，梯度幅值低。"""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    # 用高斯模糊创建软边缘
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (100, 80), (150, 180), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 15)
    # 将模糊后的掩码应用于图像
    img = img.astype(np.float32)
    dark = np.array([0, 0, 0], dtype=np.float32)
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - mask / 255.0) + dark[c] * (mask / 255.0)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_mixed_lines(size=256):
    """混合线：同时包含硬边线和软线。"""
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    # 硬边矩形
    cv2.rectangle(img, (40, 40), (90, 140), (0, 0, 0), -1)
    # 软边矩形（通过模糊实现）
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.rectangle(mask, (160, 40), (210, 140), 255, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 12)
    img_float = img.astype(np.float32)
    for c in range(3):
        img_float[:, :, c] = img_float[:, :, c] * (1 - mask / 255.0)
    return np.clip(img_float, 0, 255).astype(np.uint8)


def measure_line_width(ink, center_x, y_range=(50, 200)):
    """测量线条宽度（基于墨迹分布的半高宽）。"""
    widths = []
    for y in range(*y_range):
        line = ink[y]
        if line.sum() < 0.5:
            continue
        xs = np.arange(len(line))
        # 找到墨迹分布
        ink_region = line > 0.1
        if ink_region.sum() < 2:
            continue
        x_indices = np.where(ink_region)[0]
        width = x_indices[-1] - x_indices[0]
        widths.append(width)
    return np.mean(widths) if widths else 0


def run_benchmark():
    """运行自适应线宽基准测试。"""
    test_cases = [
        ("hard_edge", make_hard_edge_line()),
        ("soft_line", make_soft_line()),
        ("mixed_lines", make_mixed_lines()),
    ]

    results = []

    for name, img in test_cases:
        # 测试自适应模式
        t0 = time.perf_counter()
        out_adaptive = to_lineart(img, LineArtParams(adaptive_width=True))
        dt_adaptive = time.perf_counter() - t0

        # 测试非自适应模式（向后兼容）
        t0 = time.perf_counter()
        out_fixed = to_lineart(img, LineArtParams(adaptive_width=False))
        dt_fixed = time.perf_counter() - t0

        # 计算墨迹统计
        ink_adaptive = 1.0 - out_adaptive.astype(np.float64) / 255.0
        ink_fixed = 1.0 - out_fixed.astype(np.float64) / 255.0

        ink_sum_adaptive = ink_adaptive.sum()
        ink_sum_fixed = ink_fixed.sum()

        # 差异统计
        diff = np.abs(ink_adaptive - ink_fixed).mean()
        ink_diff = abs(ink_sum_adaptive - ink_sum_fixed)

        results.append({
            "name": name,
            "dt_adaptive": dt_adaptive,
            "dt_fixed": dt_fixed,
            "ink_adaptive": ink_sum_adaptive,
            "ink_fixed": ink_sum_fixed,
            "ink_diff": ink_diff,
            "mean_diff": diff,
        })

    # 输出结果
    print("=" * 80)
    print("自适应线宽基准测试结果")
    print("=" * 80)
    print(f"{'场景':<15} {'自适应时间':>12} {'固定时间':>10} {'自适应墨迹':>12} {'固定墨迹':>10} {'墨迹差异':>10} {'平均差异':>10}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<15} {r['dt_adaptive']*1000:>10.1f}ms {r['dt_fixed']*1000:>8.1f}ms "
              f"{r['ink_adaptive']:>10.0f} {r['ink_fixed']:>8.0f} "
              f"{r['ink_diff']:>8.0f} {r['mean_diff']:>8.4f}")

    print("\n说明:")
    print("- 硬边线：自适应模式应产生更窄的线宽（系数 0.4）")
    print("- 软线：自适应模式应产生更宽的线宽（系数 0.6）")
    print("- 墨迹差异：正值表示自适应模式墨迹更多")

    # 验证 backward compatibility
    print("\n向后兼容性验证:")
    blank = np.full((128, 128, 3), 255, dtype=np.uint8)
    out_old = to_lineart(blank, LineArtParams(adaptive_width=False))
    out_new = to_lineart(blank, LineArtParams(adaptive_width=True))
    print(f"  空白图输出一致: {np.allclose(out_old, out_new)}")

    return results


if __name__ == "__main__":
    run_benchmark()
