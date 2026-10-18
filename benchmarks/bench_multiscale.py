"""多尺度语义融合 vs 单尺度细线保留效果对比。

测试目标：
- 验证 multiscale=True 模式下细线保留效果
- 对比单尺度 vs 多尺度的墨迹响应差异
- 量化细线区域的保留率提升

测试方法：
1. 生成包含极细线条的测试图像
2. 分别用单尺度和多尺度模式处理
3. 对比细线区域的响应强度和保留率
"""
import sys
import time

import cv2
import numpy as np

import lineart.pipeline as PL

# 细线宽度阈值（像素）
THIN_LINE_THRESHOLD = 3

# 测试图像尺寸
TEST_SIZE = 1024


def make_thin_line_test_image(size=TEST_SIZE):
    """生成包含极细线条的测试图像。

    包含：
    - 1px 宽的水平细线
    - 1px 宽的垂直细线
    - 2px 宽的斜线
    - 3px 宽的粗线（对照）
    - 渐变背景（模拟真实场景）
    """
    img = np.ones((size, size), dtype=np.uint8) * 255

    # 渐变背景
    for i in range(size):
        bg_val = int(200 + 55 * np.sin(i / 200.0))
        img[i, :] = bg_val

    # 1px 水平细线（最难检测的）
    img[100, 50:950] = 50

    # 1px 垂直细线
    img[50:950, 200] = 50

    # 2px 斜线
    for i in range(200):
        x = 300 + i
        y = 50 + i
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 40
            if y + 1 < size:
                img[y + 1, x] = 60

    # 3px 粗线（对照）
    img[400:403, 50:950] = 30

    # 网格细线
    for i in range(50, 950, 50):
        img[i, 600:900] = 80  # 水平
        img[600:900, i] = 80  # 垂直

    # 小圆环（细线结构）
    cx, cy, r = 512, 512, 100
    for angle in np.linspace(0, 2 * np.pi, 360):
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        if 0 <= x < size and 0 <= y < size:
            img[y, x] = 60

    return img


def detect_thin_lines(edge_map, threshold=128):
    """检测边缘图中的细线区域。

    Args:
        edge_map: 边缘图，范围 [0, 255]
        threshold: 二值化阈值

    Returns:
        细线区域的二值掩码
    """
    binary = (edge_map < threshold).astype(np.uint8)

    # 计算距离变换，细线区域距离变换值较小
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)

    # 细线区域：距离变换值 <= THIN_LINE_THRESHOLD / 2
    thin_mask = (dist > 0) & (dist <= THIN_LINE_THRESHOLD / 2)

    return thin_mask


def compute_line_preservation(original, result, threshold=128):
    """计算线条保留率。

    Args:
        original: 原始测试图像
        result: 处理后的边缘图
        threshold: 二值化阈值

    Returns:
        (total_lines, preserved_lines, preservation_rate)
    """
    # 创建原始细线掩码
    # 原始图中灰度值 < 150 的区域认为是线条
    original_lines = (original < 150).astype(np.uint8)

    # 结果图中灰度值 < threshold 的区域认为是检测到的线条
    detected_lines = (result < threshold).astype(np.uint8)

    # 计算线条保留率
    total_pixels = original_lines.sum()
    preserved_pixels = (original_lines & detected_lines).sum()

    if total_pixels == 0:
        return 0, 0, 0.0

    return total_pixels, preserved_pixels, preserved_pixels / total_pixels


def compare_modes(img, coherence=0.6, detail=0.6, sensitivity=0.5):
    """对比单尺度和多尺度模式。

    Returns:
        dict: 包含各种对比指标
    """
    h, w = img.shape

    # 单尺度模式
    t0 = time.perf_counter()
    result_single = PL.fdog_semantic(
        img, coherence=coherence, detail=detail,
        sensitivity=sensitivity, multiscale=False
    )
    t_single = time.perf_counter() - t0

    # 多尺度模式
    t0 = time.perf_counter()
    result_multi = PL.fdog_semantic(
        img, coherence=coherence, detail=detail,
        sensitivity=sensitivity, multiscale=True
    )
    t_multi = time.perf_counter() - t0

    # 计算线条保留率
    total_single, preserved_single, rate_single = compute_line_preservation(img, result_single)
    total_multi, preserved_multi, rate_multi = compute_line_preservation(img, result_multi)

    # 计算墨迹强度对比
    ink_single = (result_single < 128).sum()
    ink_multi = (result_multi < 128).sum()

    # 计算细线区域的响应强度
    # 使用原始图中细线区域（灰度值 < 100 且不是粗线）
    thin_line_mask = (img < 100).astype(np.uint8)
    coarse_line_mask = (img < 50).astype(np.uint8)
    thin_line_only = thin_line_mask & (~coarse_line_mask)

    if thin_line_only.sum() > 0:
        thin_response_single = result_single[thin_line_only > 0].mean()
        thin_response_multi = result_multi[thin_line_only > 0].mean()
    else:
        thin_response_single = thin_response_multi = 0.0

    return {
        'single_time': t_single,
        'multi_time': t_multi,
        'single_ink': ink_single,
        'multi_ink': ink_multi,
        'single_preserved': preserved_single,
        'multi_preserved': preserved_multi,
        'single_total': total_single,
        'multi_total': total_multi,
        'single_rate': rate_single,
        'multi_rate': rate_multi,
        'thin_response_single': thin_response_single,
        'thin_response_multi': thin_response_multi,
        'thin_improvement': thin_response_multi - thin_response_single,
        'speedup': t_single / t_multi if t_multi > 0 else 0,
        'result_single': result_single,
        'result_multi': result_multi,
    }


def main():
    print("=" * 70)
    print("多尺度语义融合 vs 单尺度 细线保留效果对比")
    print("=" * 70)

    # 生成测试图像
    print("\n生成测试图像...")
    img = make_thin_line_test_image()
    cv2.imwrite('/tmp/bench_multiscale_input.png', img)
    print(f"测试图像尺寸: {img.shape}")
    print(f"测试图像已保存到 /tmp/bench_multiscale_input.png")

    # 对比测试
    print("\n运行对比测试...")
    result = compare_modes(img)

    # 输出结果
    print("\n" + "=" * 70)
    print("性能对比")
    print("=" * 70)
    print(f"单尺度时间: {result['single_time']*1000:.1f} ms")
    print(f"多尺度时间: {result['multi_time']*1000:.1f} ms")
    print(f"时间比: {result['speedup']:.2f}x (单尺度/多尺度)")

    print("\n" + "=" * 70)
    print("墨迹检测对比")
    print("=" * 70)
    print(f"单尺度墨迹像素数: {result['single_ink']}")
    print(f"多尺度墨迹像素数: {result['multi_ink']}")
    print(f"墨迹差异: {result['multi_ink'] - result['single_ink']:+d}")

    print("\n" + "=" * 70)
    print("线条保留率对比")
    print("=" * 70)
    print(f"单尺度总线条像素: {result['single_total']}")
    print(f"单尺度保留像素: {result['single_preserved']}")
    print(f"单尺度保留率: {result['single_rate']*100:.2f}%")
    print()
    print(f"多尺度总线条像素: {result['multi_total']}")
    print(f"多尺度保留像素: {result['multi_preserved']}")
    print(f"多尺度保留率: {result['multi_rate']*100:.2f}%")
    print()
    print(f"保留率提升: {(result['multi_rate'] - result['single_rate'])*100:+.2f}%")

    print("\n" + "=" * 70)
    print("细线响应强度对比")
    print("=" * 70)
    print(f"单尺度细线平均响应: {result['thin_response_single']:.4f}")
    print(f"多尺度细线平均响应: {result['thin_response_multi']:.4f}")
    print(f"细线响应提升: {result['thin_improvement']:+.4f}")

    # 保存结果图像
    cv2.imwrite('/tmp/bench_multiscale_single.png', result['result_single'] * 255)
    cv2.imwrite('/tmp/bench_multiscale_multi.png', result['result_multi'] * 255)
    print("\n结果图像已保存:")
    print("  /tmp/bench_multiscale_single.png (单尺度)")
    print("  /tmp/bench_multiscale_multi.png (多尺度)")

    # 输出结论
    print("\n" + "=" * 70)
    print("结论")
    print("=" * 70)
    if result['multi_rate'] > result['single_rate']:
        print(f"✓ 多尺度模式线条保留率更高，提升 {((result['multi_rate'] - result['single_rate']) * 100):.2f}%")
    else:
        print(f"✗ 单尺度模式线条保留率更高，差异 {((result['multi_rate'] - result['single_rate']) * 100):.2f}%")

    if result['thin_improvement'] > 0:
        print(f"✓ 多尺度模式细线响应更强，提升 {result['thin_improvement']:.4f}")
    else:
        print(f"✗ 单尺度模式细线响应更强，差异 {result['thin_improvement']:.4f}")

    print(f"⚠ 多尺度模式计算时间约为单尺度的 {result['speedup']:.1f}x")

    return 0


if __name__ == '__main__':
    sys.exit(main())
