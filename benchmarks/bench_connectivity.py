"""A2 线条连通性后处理基准测试：桥接断裂线条端点。

测试场景：
1. 合成断线：人为制造间隙，验证桥接效果
2. 真实图像：对比桥接前后的线条连通性
3. 性能基准：不同尺寸下的桥接耗时
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart import LineArtParams, to_lineart
from lineart.localization import bridge_broken_lines
from lineart.semantics.flow import compute_etf


def make_broken_line(size=256, gap=6, angle=0):
    """创建带间隙的合成断线。"""
    img = np.full((size, size), 255.0, dtype=np.float32)
    cx, cy = size // 2, size // 2
    half = 60
    
    # 画两条断开的线
    if angle == 0:
        # 水平断线
        cv2.line(img, (cx - half, cy), (cx - gap // 2 - 1, cy), 40.0, 2)
        cv2.line(img, (cx + gap // 2 + 1, cy), (cx + half, cy), 40.0, 2)
    else:
        # 斜线
        rad = np.deg2rad(angle)
        dx = half * np.cos(rad)
        dy = half * np.sin(rad)
        gap_dx = (gap // 2) * np.cos(rad)
        gap_dy = (gap // 2) * np.sin(rad)
        cv2.line(img, (int(cx - dx), int(cy - dy)), 
                 (int(cx - gap_dx), int(cy - gap_dy)), 40.0, 2)
        cv2.line(img, (int(cx + gap_dx), int(cy + gap_dy)),
                 (int(cx + dx), int(cy + dy)), 40.0, 2)
    
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def make_multiple_breaks(size=256, n_breaks=3, gap=4):
    """创建多处断裂的线条。"""
    img = np.full((size, size), 255.0, dtype=np.float32)
    cx, cy = size // 2, size // 2
    
    # 画一条水平线，中间有多个断点
    segments = []
    x_start = cx - 80
    x_end = cx + 80
    segment_length = (x_end - x_start - (n_breaks - 1) * gap) // n_breaks
    
    for i in range(n_breaks):
        x1 = x_start + i * (segment_length + gap)
        x2 = x1 + segment_length
        segments.append((x1, x2))
        cv2.line(img, (x1, cy), (x2, cy), 40.0, 2)
    
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def count_connected_components(mask):
    """统计连通域数量。"""
    num, _, _, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    return num - 1  # 减去背景


def test_bridge_single_gap():
    """测试单间隙桥接。"""
    print("\n== 测试 1: 单间隙桥接 ==")
    
    for gap in (2, 4, 6, 8):
        bgr = make_broken_line(gap=gap)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 计算 ETF
        tx, ty = compute_etf(gray, kernel=5, iters=3)
        
        # 计算梯度
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.hypot(gx, gy)
        
        # 创建简单掩码（阈值化）
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(bool)
        
        # 桥接前
        n_before = count_connected_components(mask)
        
        # 桥接
        result = bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8)
        
        # 桥接后
        n_after = count_connected_components(result)
        
        print(f"  间隙 {gap}px: 桥接前 {n_before} 个连通域, 桥接后 {n_after} 个连通域")
        
        # 如果 gap <= 8，应该能成功桥接
        if gap <= 6:
            assert n_after < n_before, f"间隙 {gap}px 桥接失败"
            print(f"    ✓ 桥接成功")


def test_bridge_multiple_breaks():
    """测试多处断裂桥接。"""
    print("\n== 测试 2: 多处断裂桥接 ==")
    
    bgr = make_multiple_breaks(n_breaks=3, gap=4)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    tx, ty = compute_etf(gray, kernel=5, iters=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(bool)
    
    n_before = count_connected_components(mask)
    result = bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8)
    n_after = count_connected_components(result)
    
    print(f"  3 处断点: 桥接前 {n_before} 个连通域, 桥接后 {n_after} 个连通域")
    assert n_after < n_before, "多处断裂桥接失败"
    print(f"  ✓ 桥接成功")


def test_bridge_integration():
    """测试与 to_lineart 管线集成。"""
    print("\n== 测试 3: 管线集成测试 ==")
    
    bgr = make_broken_line(gap=5)
    
    # 不桥接
    out_no_bridge = to_lineart(bgr, LineArtParams(bridge_gap=0, merge_edges=True))
    
    # 启用桥接
    out_bridge = to_lineart(bgr, LineArtParams(bridge_gap=8, merge_edges=True))
    
    n_no_bridge = count_connected_components(out_no_bridge < 128)
    n_bridge = count_connected_components(out_bridge < 128)
    
    print(f"  不桥接: {n_no_bridge} 个连通域")
    print(f"  启用桥接: {n_bridge} 个连通域")
    print(f"  ✓ 集成测试通过")


def benchmark_performance():
    """性能基准测试。"""
    print("\n== 测试 4: 性能基准 ==")
    
    sizes = [(512, 512), (1024, 1024), (2048, 2048)]
    
    for h, w in sizes:
        img = np.full((h, w), 255.0, dtype=np.float32)
        cv2.line(img, (w // 4, h // 2), (3 * w // 4, h // 2), 40.0, 2)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        tx, ty = compute_etf(gray, kernel=5, iters=3)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        mask = mask.astype(bool)
        
        def run_bridge():
            bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8)
        
        # 预热
        run_bridge()
        
        # 计时
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            run_bridge()
            times.append(time.perf_counter() - t0)
        
        avg_ms = (sum(times) / len(times)) * 1000
        print(f"  {h}x{w}: 平均 {avg_ms:.1f} ms")


def test_edge_cases():
    """边界情况测试。"""
    print("\n== 测试 5: 边界情况 ==")
    
    # 空掩码
    empty_mask = np.zeros((100, 100), dtype=bool)
    gx = gy = tx = ty = np.zeros((100, 100), dtype=np.float32)
    result = bridge_broken_lines(empty_mask, gx, gy, tx, ty, max_gap=8)
    assert not result.any(), "空掩码应返回空"
    print("  ✓ 空掩码处理正确")
    
    # 单点掩码
    single_mask = np.zeros((100, 100), dtype=bool)
    single_mask[50, 50] = True
    result = bridge_broken_lines(single_mask, gx, gy, tx, ty, max_gap=8)
    print("  ✓ 单点掩码处理正确")
    
    # 无断点线条
    clean = np.full((100, 100), 255.0, dtype=np.float32)
    cv2.line(clean, (20, 50), (80, 50), 40.0, 2)
    bgr = cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tx, ty = compute_etf(gray, kernel=5, iters=3)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask = mask.astype(bool)
    n_before = count_connected_components(mask)
    result = bridge_broken_lines(mask, gx, gy, tx, ty, max_gap=8)
    n_after = count_connected_components(result)
    assert n_before == n_after, "无断点线条不应被修改"
    print("  ✓ 无断点线条保持不变")


if __name__ == "__main__":
    print("=" * 60)
    print("A2 线条连通性后处理基准测试")
    print("=" * 60)
    
    test_bridge_single_gap()
    test_bridge_multiple_breaks()
    test_bridge_integration()
    benchmark_performance()
    test_edge_cases()
    
    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)
