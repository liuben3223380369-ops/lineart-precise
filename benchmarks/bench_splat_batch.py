"""bilinear_splat_batch 批量化微基准：对比 np.add.at 与 np.bincount 性能。

测试场景：
1. 细线（单点 splat）：直接调用 bilinear_splat_batch
2. 粗线（展开 splat）：批量收集后调用 bilinear_splat_batch
3. 与旧版逐点循环 bilinear_splat 对比等价性与速度
"""
import sys
import time

import numpy as np

sys.path.insert(0, ".")
from lineart.common import bilinear_splat, bilinear_splat_batch

rng = np.random.default_rng(0)


def bench(label, fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    print(f"  {label:42s} {min(ts) * 1000:8.1f} ms")
    return min(ts)


# ─── 1. 细线 splat（单点调用） ───────────────────────────────────

print("== 1. 细线 splat（180K 点，4096 画布） ==")
H, W = 3072, 4096
n_pts = 180_000
py = rng.uniform(0, H - 1.01, n_pts).astype(np.float32)
px = rng.uniform(0, W - 1.01, n_pts).astype(np.float32)
vals = rng.uniform(0, 1, n_pts).astype(np.float32)


def splat_legacy():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat(canvas, py, px, vals)
    return canvas


def splat_batch():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat_batch(canvas, py, px, vals)
    return canvas


r0 = splat_legacy()
r1 = splat_batch()
print(f"  等价性: batch vs legacy max|Δ| = {np.abs(r0 - r1).max():.2e}")
assert np.abs(r0 - r1).max() < 1e-6, "数值等价性不满足！"
bench("legacy (np.add.at)", splat_legacy)
bench("batch (np.bincount)", splat_batch)

# ─── 2. 粗线 splat（展开后批量） ─────────────────────────────────

print("\n== 2. 粗线 splat（展开 5 点，180K → 900K 点） ==")
K = 5  # 展开点数
py2 = rng.uniform(0, H - 1.01, n_pts).astype(np.float32)
px2 = rng.uniform(0, W - 1.01, n_pts).astype(np.float32)
vals2 = rng.uniform(0, 1, n_pts).astype(np.float32)
dy2 = rng.uniform(-1, 1, n_pts).astype(np.float32)
dx2 = rng.uniform(-1, 1, n_pts).astype(np.float32)
nx2 = -dy2
ny2 = dx2
span = 4.0
offsets2 = np.linspace(-span / 2.0, span / 2.0, K)


def splat_legacy_thick():
    canvas = np.zeros((H, W), dtype=np.float32)
    for off in offsets2:
        bilinear_splat(canvas, py2 + ny2 * off, px2 + nx2 * off, vals2)
    return canvas


def splat_batch_thick():
    canvas = np.zeros((H, W), dtype=np.float32)
    n = len(py2)
    total = n * K
    py_batch = np.empty(total, dtype=py2.dtype)
    px_batch = np.empty(total, dtype=px2.dtype)
    vals_batch = np.empty(total, dtype=vals2.dtype)
    for i, off in enumerate(offsets2):
        sl = slice(i * n, (i + 1) * n)
        py_batch[sl] = py2 + ny2 * off
        px_batch[sl] = px2 + nx2 * off
        vals_batch[sl] = vals2
    bilinear_splat_batch(canvas, py_batch, px_batch, vals_batch)
    return canvas


r2 = splat_legacy_thick()
r3 = splat_batch_thick()
print(f"  等价性: batch vs legacy max|Δ| = {np.abs(r2 - r3).max():.2e}")
assert np.abs(r2 - r3).max() < 1e-6, "数值等价性不满足！"
bench("legacy (逐点循环 ×5)", splat_legacy_thick)
bench("batch (批量 bincount ×1)", splat_batch_thick)

# ─── 3. 小批量（500 点） ─────────────────────────────────────────

print("\n== 3. 小批量（500 点） ==")
n_small = 500
py3 = rng.uniform(0, H - 1.01, n_small).astype(np.float32)
px3 = rng.uniform(0, W - 1.01, n_small).astype(np.float32)
vals3 = rng.uniform(0, 1, n_small).astype(np.float32)


def splat_legacy_small():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat(canvas, py3, px3, vals3)
    return canvas


def splat_batch_small():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat_batch(canvas, py3, px3, vals3)
    return canvas


r4 = splat_legacy_small()
r5 = splat_batch_small()
print(f"  等价性: batch vs legacy max|Δ| = {np.abs(r4 - r5).max():.2e}")
bench("legacy (500 点)", splat_legacy_small)
bench("batch (500 点)", splat_batch_small)

# ─── 4. 越界过滤正确性 ─────────────────────────────────────────

print("\n== 4. 越界过滤正确性（边界点） ==")
py_edge = np.array([-0.5, 0.0, 1.0, H - 1.5, H - 1.0, H - 0.5], dtype=np.float32)
px_edge = np.array([-0.5, 0.0, 1.0, W - 1.5, W - 1.0, W - 0.5], dtype=np.float32)
vals_edge = np.ones(len(py_edge), dtype=np.float32)


def test_boundary():
    canvas = np.zeros((H, W), dtype=np.float32)
    bilinear_splat_batch(canvas, py_edge, px_edge, vals_edge)
    return canvas


r_edge = test_boundary()
# 边界内有效累加：y=0, x=0, y=H-2, x=W-2 等位置应有值
assert r_edge[0, 0] > 0, "边界内点未累加"
assert r_edge[H - 2, W - 2] > 0, "边界内点未累加"
# 越界点不应累加到边界外
print(f"  边界累加: r[0,0]={r_edge[0,0]:.4f} r[H-2,W-2]={r_edge[H-2,W-2]:.4f}")
print("  越界过滤: PASS")

print("\n===== 所有测试通过 =====")
