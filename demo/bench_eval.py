#!/usr/bin/env python3
"""多风格本地模型评测（v1.27.2 调试工具）。

对 demo/bench/ 全部图片跑四语义（fdog / neural / dexined / ensemble）：
- 合成图（*_gt.png 真值）：召回 / 精确率 / F1（输出线 vs 真值线，容差 2px 膨胀）
- 真实图（无真值）：结构覆盖（输出线对 Canny 结构边缘的召回）、
  平坦区噪率（输出线落在局部平坦区域的占比）、墨点占比、连通碎片度

用法：
  python3 demo/bench_eval.py                 # 全部
  python3 demo/bench_eval.py ink anime       # 文件名子串过滤
输出：demo/bench/report.md + stdout 表格
"""
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from lineart import LineArtParams, to_lineart  # noqa: E402

BENCH = os.path.join(os.path.dirname(__file__), "bench")
SEMANTICS = ("fdog", "neural", "dexined", "ensemble")
DIL = 2  # 真值容差膨胀半径


def load_gt(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return g if g is not None else None


def out_ink(result):
    """输出图 → ink 图（0=白 1=黑线，float）。"""
    g = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if result.ndim == 3 else result
    return (255.0 - g.astype(np.float32)) / 255.0


def eval_with_gt(ink, gt):
    """召回 = 膨胀预测覆盖真值比例；精确率 = 预测落在膨胀真值内比例。"""
    gt_mask = gt > 127
    pred = ink > 0.5
    gt_d = cv2.dilate(gt_mask.astype(np.uint8),
                      np.ones((2 * DIL + 1,) * 2, np.uint8)) > 0
    pred_d = cv2.dilate(pred.astype(np.uint8),
                        np.ones((2 * DIL + 1,) * 2, np.uint8)) > 0
    recall = (pred_d & gt_mask).sum() / max(gt_mask.sum(), 1)
    precision = (pred & gt_d).sum() / max(pred.sum(), 1)
    f1 = 2 * recall * precision / max(recall + precision, 1e-9)
    return {"recall": recall, "precision": precision, "f1": f1}


def eval_structure(ink, gray):
    """真实图结构指标：Canny 结构边缘召回 + 平坦区噪率 + 墨点 + 碎片度。"""
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 120) > 0
    pred = ink > 0.5
    edges_d = cv2.dilate(edges.astype(np.uint8),
                         np.ones((5, 5), np.uint8)) > 0
    struct_recall = (pred & edges_d).sum() / max(edges_d.sum(), 1)
    # 平坦区：局部梯度幅值低
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.dilate(np.hypot(gx, gy), np.ones((5, 5), np.uint8))
    flat = mag < 8.0
    noise_on_flat = (pred & flat).sum() / max(pred.sum(), 1)
    ink_ratio = pred.mean()
    n_ink = int(pred.sum())
    frag = 0.0
    if n_ink > 0:
        n_lab, _ = cv2.connectedComponents(pred.astype(np.uint8), 8)
        frag = (n_lab - 1) / max(n_ink / 1000.0, 1e-9)  # 每千墨点连通域数
    return {"struct_recall": struct_recall, "noise_flat": noise_on_flat,
            "ink_ratio": ink_ratio, "frag_per_kpx": frag}


def main():
    args = sys.argv[1:]
    files = sorted(f for f in os.listdir(BENCH)
                   if f.endswith((".png", ".jpg"))
                   and "_gt" not in f and not f.startswith("out_"))
    if args:
        files = [f for f in files if any(a in f for a in args)]
    rows = []
    for fname in files:
        path = os.path.join(BENCH, fname)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"!! 跳过无法读取: {fname}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gt = load_gt(path.rsplit(".", 1)[0] + "_gt.png")
        stem = fname.rsplit(".", 1)[0]
        for sem in SEMANTICS:
            t0 = time.perf_counter()
            try:
                result = to_lineart(img, LineArtParams(semantic=sem))
            except Exception as e:
                print(f"!! {stem} {sem}: 异常 {type(e).__name__}: {e}")
                rows.append({"file": stem, "semantic": sem,
                             "error": f"{type(e).__name__}: {e}"})
                continue
            dt = time.perf_counter() - t0
            ink = out_ink(result)
            row = {"file": stem, "semantic": sem, "sec": round(dt, 1)}
            if gt is not None:
                row.update(eval_with_gt(ink, gt))
            else:
                row.update(eval_structure(ink, gray))
            # 输出线稿存档（人工复核用）
            out_path = os.path.join(BENCH, f"out_{stem}_{sem}.png")
            cv2.imwrite(out_path, result)
            rows.append(row)
            if gt is not None:
                print(f"{stem:22s} {sem:9s} R={row['recall']:.3f} "
                      f"P={row['precision']:.3f} F1={row['f1']:.3f} "
                      f"({row['sec']}s)")
            else:
                print(f"{stem:22s} {sem:9s} 结构={row['struct_recall']:.3f} "
                      f"噪率={row['noise_flat']:.3f} 墨占={row['ink_ratio']:.3f} "
                      f"碎片={row['frag_per_kpx']:.1f} ({row['sec']}s)")
    # 报告
    report = os.path.join(BENCH, "report.md")
    with open(report, "w", encoding="utf-8") as f:
        f.write("# 多风格本地模型评测报告\n\n")
        f.write(f"图 {len(files)} 张 × 语义 {len(SEMANTICS)} 档；"
                f"容差 {DIL}px；时间 {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        for sem in SEMANTICS:
            f.write(f"\n## {sem}\n\n")
            sub = [r for r in rows if r["semantic"] == sem]
            if any("recall" in r for r in sub):
                f.write("| 图 | 召回 | 精确率 | F1 | 耗时s |\n|---|---|---|---|---|\n")
                for r in sub:
                    if "recall" in r:
                        f.write(f"| {r['file']} | {r['recall']:.3f} | "
                                f"{r['precision']:.3f} | {r['f1']:.3f} | "
                                f"{r['sec']} |\n")
            if any("struct_recall" in r for r in sub):
                f.write("\n| 图 | 结构覆盖 | 噪率 | 墨占比 | 碎片/kpx | 耗时s |\n"
                        "|---|---|---|---|---|---|\n")
                for r in sub:
                    if "struct_recall" in r:
                        f.write(f"| {r['file']} | {r['struct_recall']:.3f} | "
                                f"{r['noise_flat']:.3f} | {r['ink_ratio']:.3f} | "
                                f"{r['frag_per_kpx']:.1f} | {r['sec']} |\n")
            errs = [r for r in sub if "error" in r]
            for r in errs:
                f.write(f"\n- ⚠ {r['file']}: {r['error']}\n")
    print(f"\n报告: {report}")


if __name__ == "__main__":
    main()
