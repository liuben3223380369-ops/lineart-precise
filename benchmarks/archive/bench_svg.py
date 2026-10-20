# -*- coding: utf-8 -*-
"""
文件名: bench_svg.py
描述: SVG 导出性能基准测试：raster_to_svg 与 to_svg_lineart 的吞吐量与简化容差对比
"""
import sys
import time
import os

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from lineart import LineArtParams, to_lineart, to_svg_lineart, raster_to_svg, save_svg


def bench(label, fn, n=5):
    """微基准测试：返回最小耗时（ms）。"""
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(f"  {label:42s} {min(ts) * 1000:8.1f} ms")
    return min(ts)


def make_test_mask(size=512, n_lines=20):
    """生成测试用二值掩码：白底黑线，含多条随机线条。"""
    mask = np.full((size, size), 255, dtype=np.uint8)
    rng = np.random.default_rng(42)
    for _ in range(n_lines):
        y1, x1 = rng.integers(0, size, 2)
        y2, x2 = rng.integers(0, size, 2)
        cv2.line(mask, (x1, y1), (x2, y2), 0, 1 + rng.integers(0, 3))
    # 添加一些圆形
    for _ in range(5):
        cx, cy = rng.integers(0, size, 2)
        r = rng.integers(10, 60)
        cv2.circle(mask, (cx, cy), r, 0, 1)
    return mask


def make_test_bgr(size=512):
    """生成测试用 BGR 图像。"""
    mask = make_test_mask(size, n_lines=30)
    # 转为灰度线稿图
    gray = mask.copy()
    # 添加一些噪声模拟真实线稿
    rng = np.random.default_rng(123)
    gray = gray.astype(np.float32) + rng.normal(0, 5, gray.shape)
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr


def main():
    print("=" * 60)
    print("SVG 导出性能基准测试")
    print("=" * 60)

    # 准备测试数据
    mask = make_test_mask(512, n_lines=30)
    bgr = make_test_bgr(512)

    # 1. raster_to_svg 不同简化容差对比
    print("\n== 1. raster_to_svg 简化容差对比（512x512 掩码） ==")

    for tol in [0.5, 1.0, 2.0, 5.0]:
        fn = lambda t=tol: raster_to_svg(mask, simplify_tol=t)
        bench(f"  simplify_tol={tol}", fn, n=3)

    # 2. SVG 输出大小对比
    print("\n== 2. SVG 输出大小对比 ==")
    for tol in [0.5, 1.0, 2.0, 5.0]:
        svg = raster_to_svg(mask, simplify_tol=tol)
        print(f"  simplify_tol={tol}: {len(svg)} bytes, {svg.count('<path')} paths")

    # 3. to_svg_lineart 端到端测试
    print("\n== 3. to_svg_lineart 端到端测试（512x512 BGR） ==")

    params = LineArtParams()
    bench("  to_svg_lineart (default)", lambda: to_svg_lineart(bgr, params), n=3)

    # 4. 不同尺寸测试
    print("\n== 4. 不同尺寸测试 ==")
    for size in [256, 512, 1024]:
        m = make_test_mask(size, n_lines=size // 16)
        svg = raster_to_svg(m, simplify_tol=1.0)
        print(f"  {size}x{size}: {len(svg) // 1024} KB, {svg.count('<path')} paths")

    # 5. 空掩码测试
    print("\n== 5. 空掩码边界测试 ==")
    empty_mask = np.full((256, 256), 255, dtype=np.uint8)
    svg = raster_to_svg(empty_mask, simplify_tol=1.0)
    print(f"  空掩码: {len(svg)} bytes")
    assert "white" in svg, "空掩码应生成白色背景"

    # 6. 保存测试
    print("\n== 6. 保存测试 ==")
    svg = raster_to_svg(mask, simplify_tol=1.0)
    dst = "/tmp/test_lineart.svg"
    save_svg(svg, dst)
    print(f"  已保存: {dst}")

    # 验证保存的 SVG 内容
    with open(dst, "r") as f:
        content = f.read()
    assert "<svg" in content, "SVG 应包含 svg 标签"
    assert "<path" in content, "SVG 应包含 path 标签"
    print(f"  验证通过: {len(content)} bytes")

    print("\n" + "=" * 60)
    print("基准测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
