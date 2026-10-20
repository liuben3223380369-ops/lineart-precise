"""work_scale 0.35 vs 0.25 真实样本差异性质分析。

问题：0.25 在真实 CAD 样本上 IoU 偏低（0.78~0.99），差异是
  (a) 边缘阈值敏感 —— 差异像素紧贴线边缘 1~3px，线没丢
  (b) 结构性丢线   —— 差异像素远离对方版本的线，成块丢失
判定：对单向差异做距离变换，dist<=TOL 归为边缘带，dist>TOL 归为孤立；
     孤立部分做连通域分析。另算容差 IoU（tol-IoU）作总括指标。
"""
import sys
import time
import cv2
import numpy as np

import lineart.pipeline as PL

SAMPLES = ['blueprint', 'cad_black_color', 'cad_gray_bg', 'cad_grid_bg',
           'cad_thick_lines', 'cad_with_text']
TARGET = 4096
TOL = 3


def convert_at(img, ws):
    orig = PL.fdog_semantic

    def patched(gray, **kw):
        kw['work_scale'] = ws
        return orig(gray, **kw)

    PL.fdog_semantic = patched
    try:
        return PL.to_lineart(img)
    finally:
        PL.fdog_semantic = orig


def analyze(name, img):
    h, w = img.shape[:2]
    s = TARGET / max(h, w)
    big = cv2.resize(img, (int(round(w * s)), int(round(h * s))),
                     interpolation=cv2.INTER_CUBIC)
    t0 = time.perf_counter()
    out35 = convert_at(big, 0.35)
    out25 = convert_at(big, 0.25)
    dt = time.perf_counter() - t0

    a = out35 < 128    # 0.35 的墨
    b = out25 < 128    # 0.25 的墨
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    iou = inter / max(union, 1)

    # 容差 IoU：b 的每个像素须落在 a 的 TOL 邻域内
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * TOL + 1, 2 * TOL + 1))
    a_tol = cv2.dilate(a.astype(np.uint8), k) > 0
    iou_tol = np.logical_and(a_tol, b).sum() / max(b.sum(), 1)

    def one_way(d_only, other):
        """d_only: 本版独有墨; other: 对方版本的墨. 返回 (near, far, far_ncc, far_max)."""
        if d_only.sum() == 0:
            return 0, 0, 0, 0
        dist = cv2.distanceTransform((~other).astype(np.uint8),
                                     cv2.DIST_L2, 3)
        d = dist[d_only]
        far_mask = np.zeros_like(other, dtype=np.uint8)
        far_mask[d_only & (dist > TOL)] = 1
        n, lab, stats, _ = cv2.connectedComponentsWithStats(far_mask, 8)
        areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([])
        return int((d <= TOL).sum()), int((d > TOL).sum()), \
            int(len(areas)), int(areas.max()) if len(areas) else 0

    d35 = a & ~b     # 0.25 丢失的墨
    d25 = b & ~a     # 0.25 新增的墨
    n35, f35, cc35, mx35 = one_way(d35, b)
    n25, f25, cc25, mx25 = one_way(d25, a)

    print(f"{name:18s} ink35={a.sum():9d} ink25={b.sum():9d} "
          f"IoU={iou:.4f} tolIoU={iou_tol:.4f} t={dt:.1f}s")
    print(f"  丢失(35有25无): 边缘带={n35:8d} 孤立={f35:8d} "
          f"孤立CC数={cc35:4d} 最大CC={mx35:7d}")
    print(f"  新增(25有35无): 边缘带={n25:8d} 孤立={f25:8d} "
          f"孤立CC数={cc25:4d} 最大CC={mx25:7d}")
    sys.stdout.flush()
    return iou, iou_tol


def main():
    rows = []
    for name in SAMPLES:
        img = cv2.imread(f'demo/samples/{name}.png')
        rows.append((name,) + analyze(name, img))
    print("\n=== 汇总 ===")
    for name, iou, iou_tol in rows:
        verdict = "结构风险" if iou_tol < 0.98 else "边缘敏感"
        print(f"{name:18s} IoU={iou:.4f} tolIoU={iou_tol:.4f} {verdict}")


if __name__ == '__main__':
    main()
