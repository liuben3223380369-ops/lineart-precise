"""第 8 轮改动量化验证（修复版）。"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, "/workspace/lineart-precise")
sys.path.insert(0, "/workspace/lineart-precise/tests")
from lineart import LineArtParams, to_lineart
from lineart.shock import coherence_shock
from test_precision import edge_position_vertical, make_circle, make_subpixel_edge


def run_ink(img_f, **kw):
    bgr = cv2.cvtColor(np.clip(img_f, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    out = to_lineart(bgr, LineArtParams(**kw))
    return 1.0 - out.astype(np.float64) / 255.0


print("=" * 70)
print("1. 细线亚像素精度（thickness=1 走 1x 路径，应回到基线）")
print("=" * 70)
for label, kw in [("v1.3.0 基线", dict(render_ss=1)),
                  ("ss=2（细线自动 1x）", dict(render_ss=2))]:
    errs, spreads = [], []
    for ex in (64.0, 64.25, 64.5, 64.75):
        ink = run_ink(make_subpixel_edge(edge_x=ex), **kw)
        mean, spread = edge_position_vertical(ink)
        errs.append(mean - ex)
        spreads.append(spread)
    print(f"  {label:<22} 平均|误差| {np.mean(np.abs(errs)):.4f}px  最大 {np.max(np.abs(errs)):.4f}px  散布 {np.mean(spreads):.4f}px")

print()
print("=" * 70)
print("2. 粗线渲染（thickness=3）：SSR vs 1x 的宽度准确性与对称性")
print("=" * 70)
for label, kw in [("1x", dict(render_ss=1, thickness=3.0)),
                  ("SSR=2", dict(render_ss=2, thickness=3.0))]:
    ink = run_ink(make_subpixel_edge(edge_x=64.4), **kw)
    row = ink[64]
    lo, hi = row.min(), row.max()
    n = (row - lo) / max(hi - lo, 1e-9)
    xs = np.arange(len(n)) + 0.5
    # 上升沿（暗→亮的墨迹上升）：ink 从 0 到 1
    x10 = xs[n >= 0.1][0]
    x90 = xs[n >= 0.9][0]
    prof = row / max(row.max(), 1e-9)
    xc = (prof * xs).sum() / prof.sum()
    print(f"  {label:<6} 线宽(10-90%) {x90-x10:.3f}px（目标 3px）  质心 {xc:.3f}（目标 64.4）  峰值 {hi:.3f}")

print()
print("=" * 70)
print("3. shock 锐化：单元收窄 + 管线端到端位置保持（检测/定位分离）")
print("=" * 70)
sharp_edge = make_subpixel_edge(edge_x=64.3, bg_dark=True) / 255.0
blur_edge = cv2.GaussianBlur(sharp_edge, (0, 0), 1.6)

def edge_pos_of(gray):
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    pos = []
    for y in range(16, 112):
        row = np.abs(gx[y])
        if row.sum() < 1e-6:
            continue
        xs = np.arange(len(row)) + 0.5
        pos.append((row * xs).sum() / row.sum())
    return float(np.mean(pos))

def fall_width10_90(prof):
    """下降沿（左亮右暗）过渡带宽度。"""
    lo, hi = prof.min(), prof.max()
    n = (hi - prof) / max(hi - lo, 1e-9)  # 归一化为上升形状
    xs = np.arange(len(n)) + 0.5
    return float(xs[n >= 0.9][0] - xs[n >= 0.1][0])

p_true, p_blur = edge_pos_of(sharp_edge), edge_pos_of(blur_edge)
shocked = coherence_shock(blur_edge, iterations=4)
w_blur = fall_width10_90(blur_edge[64])
w_shock = fall_width10_90(shocked[64])
print(f"  [单元] 边缘位置: 真值 {p_true:.4f} | 模糊 {p_blur:.4f}")
print(f"  [单元] 过渡带: 模糊 {w_blur:.2f}px → shock {w_shock:.2f}px（收窄 {w_blur/max(w_shock,1e-3):.1f}x）")
print(f"  [单元] 注: 冲击会把中间灰度推向 0/1（亚像素相位量化到网格），")
print(f"         故管线中锐化图仅用于检测/强度，拟合场保留原图（见下）")

errs_on, errs_off = [], []
for ex in (64.0, 64.25, 64.5, 64.75):
    img = cv2.GaussianBlur(make_subpixel_edge(edge_x=ex, bg_dark=True), (0, 0), 1.6)
    for tag, sharpen, acc in (("off", 0.0, errs_off), ("on", 0.6, errs_on)):
        ink = run_ink(img, sharpen=sharpen)
        mean, _ = edge_position_vertical(ink)
        acc.append(mean - ex)
print(f"  [管线] 模糊边端到端精度: sharpen=0 平均|误差| {np.mean(np.abs(errs_off)):.4f}px")
print(f"  [管线] 模糊边端到端精度: sharpen=0.6 平均|误差| {np.mean(np.abs(errs_on)):.4f}px（目标: 与 off 差 <0.03px）")

print()
print("=" * 70)
print("4. 管线端到端：模糊圆输入 + sharpen")
print("=" * 70)
circ_blur = cv2.GaussianBlur(make_circle(radius=40.3, noise=0.02), (0, 0), 1.8)
for label, kw in [("sharpen=0", dict(sharpen=0.0)), ("sharpen=0.6", dict(sharpen=0.6))]:
    ink = run_ink(circ_blur, **kw)
    ys, xs = np.nonzero(ink > 0.3)
    d = np.sqrt((ys - 96.2) ** 2 + (xs - 96.7) ** 2)
    w_ = ink[ys, xs]
    r_mean = (d * w_).sum() / w_.sum()
    r_std = np.sqrt(((d - r_mean) ** 2 * w_).sum() / w_.sum())
    print(f"  {label:<12} 半径误差 {abs(r_mean-40.3):.4f}px  径向散布 {r_std:.4f}px  墨迹均值 {ink.mean():.4f}")

print()
print("=" * 70)
print("5. consensus 滤波单独效果（噪声圆，on/off 真实对比）")
print("=" * 70)
for label, kw in [("consensus on", dict(consensus=True)),
                  ("consensus off", dict(consensus=False))]:
    ink = run_ink(make_circle(noise=0.04), **kw)
    ys, xs = np.nonzero(ink > 0.3)
    d = np.sqrt((ys - 96.2) ** 2 + (xs - 96.7) ** 2)
    w_ = ink[ys, xs]
    r_mean = (d * w_).sum() / w_.sum()
    r_std = np.sqrt(((d - r_mean) ** 2 * w_).sum() / w_.sum())
    print(f"  {label:<14} 半径误差 {abs(r_mean-40.3):.4f}px  径向散布 {r_std:.4f}px")

print()
print("=" * 70)
print("6. 性能")
print("=" * 70)
img = cv2.imread("/workspace/lineart-precise/demo/samples/sketch_photo.png", cv2.IMREAD_COLOR)
small = cv2.resize(img, (1600, 1200), interpolation=cv2.INTER_AREA)
big = cv2.resize(img, (4096, 3072), interpolation=cv2.INTER_CUBIC)
for name, im in [("1600x1200", small), ("4096x3072", big)]:
    for label, kw in [("ss=1", dict(render_ss=1)), ("ss=2", dict(render_ss=2)), ("ss=2+shk.6", dict(render_ss=2, sharpen=0.6))]:
        to_lineart(im, LineArtParams(**kw))
        t0 = time.perf_counter()
        to_lineart(im, LineArtParams(**kw))
        print(f"  {name} {label:<10} {(time.perf_counter()-t0)*1000:7.1f} ms")
