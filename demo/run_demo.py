"""演示：生成合成样例 -> 运行管线 -> 输出对比与指标。"""
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from lineart import LineArtParams, to_lineart

SAMPLES = os.path.join(os.path.dirname(__file__), "samples")
OUTPUT = os.path.join(os.path.dirname(__file__), "output")

SUBPIXEL_TRUTH = {40.0: (10, 70), 60.25: (80, 150), 80.5: (10, 70), 100.75: (80, 150)}


def measure_strip(ink):
    print("\n--- 亚像素条带定位（真值 x = 40 / 60.25 / 80.5 / 100.75）---")
    rows_report = []
    for edge_x, (y0, y1) in SUBPIXEL_TRUTH.items():
        lo = max(0, int(edge_x) - 6)
        hi = min(ink.shape[1], int(edge_x) + 7)
        centroids = []
        for y in range(y0, y1):
            win = ink[y, lo:hi]
            if win.sum() < 0.5:
                continue
            xs = np.arange(lo, hi, dtype=np.float64) + 0.5
            centroids.append((win * xs).sum() / win.sum())
        c = np.mean(centroids)
        err = c - edge_x
        rows_report.append(abs(err))
        print(f"  真值 {edge_x:>7.2f}  实测 {c:>7.3f}  误差 {err:+.3f} px")
    print(f"  平均绝对误差: {np.mean(rows_report):.3f} px")
    return np.mean(rows_report)


def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from make_cad_samples import main as make_cad
    from make_scene_samples import main as make_scenes
    from make_test_image import main as make_samples
    os.makedirs(OUTPUT, exist_ok=True)
    make_samples()
    make_cad()
    make_scenes()

    configs = [
        ("antialiased", LineArtParams(mode="antialiased")),
        ("binary", LineArtParams(mode="binary", min_area=12)),
        ("xdog", LineArtParams(semantic="xdog")),
    ]
    for name, params in configs:
        for fname in sorted(os.listdir(SAMPLES)):
            src = os.path.join(SAMPLES, fname)
            img = cv2.imread(src, cv2.IMREAD_COLOR)
            t0 = time.time()
            out = to_lineart(img, params)
            dt = (time.time() - t0) * 1000
            dst = os.path.join(OUTPUT, f"{os.path.splitext(fname)[0]}_{name}.png")
            cv2.imwrite(dst, out)
            ink = 1.0 - out.astype(np.float64) / 255.0
            print(f"[{name:<12}] {fname:<22} {img.shape[1]}x{img.shape[0]:<6}"
                  f" {dt:6.0f} ms  墨迹占比 {np.mean(ink > 0.5):6.2%}")

    strip = cv2.imread(os.path.join(OUTPUT, "subpixel_strip_antialiased.png"),
                       cv2.IMREAD_GRAYSCALE)
    measure_strip(1.0 - strip.astype(np.float64) / 255.0)

    print("\n--- 场景自适应（scene=auto）---")
    from lineart.scenes import detect_scene
    for fname in sorted(os.listdir(SAMPLES)):
        img = cv2.imread(os.path.join(SAMPLES, fname), cv2.IMREAD_COLOR)
        info = detect_scene(img)
        out = to_lineart(img, LineArtParams())
        dst = os.path.join(OUTPUT, f"scene_{fname}")
        cv2.imwrite(dst, out)
        ink = 1.0 - out.astype(np.float64) / 255.0
        print(f"  {fname:<22} -> {info.scene:<10} conf={info.confidence:.2f}"
              f"  墨迹 {np.mean(ink > 0.5):5.2%}")

    print(f"\n输出目录: {OUTPUT}")


if __name__ == "__main__":
    main()
