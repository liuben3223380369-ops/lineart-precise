"""shape 外形保真路线 vs splat 中心线重绘路线 对比基准。

指标：
1. 均匀 3px / 10px 硬边竖线：外形 IoU、输出线宽、边缘过渡带宽（10-90%）
2. 变宽线（2→8px 线性渐变）：逐段宽度保真（相关系数 + 平均宽度误差）
3. 模糊圆（r=40）：径向误差（中心位置精准度）
4. 4096 密集线稿：端到端耗时

墨判定：输出 < 128 为墨（背景为白）。
"""
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, ".")
from lineart import LineArtParams, to_lineart

SPLAT = LineArtParams()
SHAPE = LineArtParams(render_mode="shape")


def vline(width_px, h=192, w=192, cx=96.0, color=40):
    img = np.full((h, w, 3), 245, np.uint8)
    x0 = int(round(cx - width_px / 2))
    x1 = int(round(cx + width_px / 2))
    gt = np.zeros((h, w), bool)
    gt[:, max(x0, 0):min(x1, w)] = True
    img[gt] = color
    return img, gt


def taper_line(h=192, w=192, cx=96.0, w0=2.0, w1=8.0, color=40):
    """宽度从 w0（顶部）线性变到 w1（底部）的竖线。"""
    img = np.full((h, w, 3), 245, np.uint8)
    gt = np.zeros((h, w), bool)
    for y in range(h):
        t = y / (h - 1)
        width = w0 + (w1 - w0) * t
        half = width / 2.0
        x_lo = int(np.ceil(cx - half))
        x_hi = int(np.floor(cx + half))
        if x_hi < x_lo:
            continue
        gt[y, max(x_lo, 0):min(x_hi + 1, w)] = True
    img[gt] = color
    return img, gt


def blurred_circle(h=192, w=192, r=40.0, color=60, sigma=1.5):
    img = np.full((h, w, 3), 245, np.uint8)
    cv2.circle(img, (w // 2, h // 2), int(r), (color,) * 3, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), sigma)
    return img, (w / 2.0, h / 2.0, r)


def ink_mask(out):
    return out < 128


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / union if union else 0.0


def width_profile(mask, axis=1):
    """逐行墨宽。"""
    return mask.sum(axis=axis).astype(np.float64)


def edge_transition(out, row=96):
    """单行墨度剖面 10-90% 过渡带宽（左右边缘合计）。"""
    prof = 1.0 - out[row].astype(np.float64) / 255.0  # 归一化墨度
    lo, hi = prof.min(), prof.max()
    if hi <= lo + 1e-6:
        return 0.0
    prof = (prof - lo) / (hi - lo)
    xs = np.where(prof > 0.01)[0]
    if len(xs) < 4:
        return 0.0
    # 左边缘：0.1→0.9 的 x 跨度；右边缘同理
    left = xs[0]
    right = xs[-1]
    def span(x_from, x_to, asc):
        seg = prof[x_from:x_to + 1] if asc else prof[x_to:x_from - 1:-1]
        t = np.linspace(0, 1, len(seg))
        x10 = np.interp(0.1, seg, t) * (len(seg) - 1)
        x90 = np.interp(0.9, seg, t) * (len(seg) - 1)
        return abs(x90 - x10)
    return span(left, left + 30, True) + span(right - 30, right, False)


def radial_error(out, cx, cy, r):
    m = ink_mask(out)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return float("inf"), float("inf")
    d = np.hypot(xs - cx, ys - cy)
    return abs(d.mean() - r), d.std()


def bench_uniform(width):
    img, gt = vline(width)
    row = img.shape[0] // 2
    res = {}
    for name, p in (("splat", SPLAT), ("shape", SHAPE)):
        out = to_lineart(img, p)
        m = ink_mask(out)
        res[name] = {
            "iou": iou(m, gt),
            "width": width_profile(m)[row],
            "trans": edge_transition(out, row),
        }
    return res, width


def bench_taper():
    img, gt = taper_line()
    gt_prof = width_profile(gt)
    res = {}
    for name, p in (("splat", SPLAT), ("shape", SHAPE)):
        out = to_lineart(img, p)
        prof = width_profile(ink_mask(out))
        if len(prof) == len(gt_prof):
            corr = float(np.corrcoef(prof, gt_prof)[0, 1])
            mae = float(np.abs(prof - gt_prof).mean())
        else:
            corr, mae = 0.0, float("inf")
        res[name] = {"corr": corr, "mae": mae,
                     "w_top": prof[8], "w_bot": prof[-8]}
    return res, (gt_prof[8], gt_prof[-8])


def bench_circle():
    img, (cx, cy, r) = blurred_circle()
    res = {}
    for name, p in (("splat", SPLAT), ("shape", SHAPE)):
        out = to_lineart(img, p)
        err, std = radial_error(out, cx, cy, r)
        res[name] = {"radial_err": err, "std": std}
    return res


def bench_perf():
    rng = np.random.default_rng(42)
    h, w = 2048, 2048
    img = np.full((h, w, 3), 245, np.uint8)
    for _ in range(80):
        x, y = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        cv2.line(img, (x, y), (x2, y2), (35,) * 3, int(rng.integers(1, 6)),
                 cv2.LINE_AA)
    for name, p in (("splat", SPLAT), ("shape", SHAPE)):
        t0 = time.perf_counter()
        to_lineart(img, p)
        res = time.perf_counter() - t0
        print(f"  2048x2048 {name:6s} {res*1000:7.0f} ms")
        globals()[f"_t_{name}"] = res


if __name__ == "__main__":
    print("=" * 72)
    print("1. 均匀硬边竖线（外形 IoU / 输出线宽 / 边缘过渡带宽 10-90%）")
    print("=" * 72)
    for width in (3, 10):
        res, w0 = bench_uniform(width)
        print(f"  [{width}px 线] 真值宽 {w0}")
        for name in ("splat", "shape"):
            r = res[name]
            print(f"    {name:6s} IoU {r['iou']:.3f}  宽 {r['width']:.1f}px  "
                  f"过渡带 {r['trans']:.2f}px")
    print()
    print("=" * 72)
    print("2. 变宽线 2→8px（宽度保真：相关系数 / 平均宽度误差）")
    print("=" * 72)
    res, (wt, wb) = bench_taper()
    print(f"  真值宽度: 顶 {wt:.1f}px → 底 {wb:.1f}px")
    for name in ("splat", "shape"):
        r = res[name]
        print(f"    {name:6s} corr {r['corr']:.3f}  MAE {r['mae']:.2f}px  "
              f"(输出 顶 {r['w_top']:.1f} → 底 {r['w_bot']:.1f})")
    print()
    print("=" * 72)
    print("3. 模糊圆 r=40（径向误差 / 径向散布）")
    print("=" * 72)
    res = bench_circle()
    for name in ("splat", "shape"):
        r = res[name]
        print(f"    {name:6s} 径向误差 {r['radial_err']:.4f}px  散布 {r['std']:.4f}px")
    print()
    print("=" * 72)
    print("4. 性能（2048 密集线稿）")
    print("=" * 72)
    bench_perf()
