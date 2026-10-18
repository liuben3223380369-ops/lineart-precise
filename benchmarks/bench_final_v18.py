"""v1.8.0 终验：当前代码 vs v1.7.1 基线在 4096 大图上的输出一致性。

用法:
  python3 benchmarks/bench_final_v18.py save new   # 当前代码输出 -> /tmp/v18_new_*.npy
  (git stash 切回基线后)
  python3 benchmarks/bench_final_v18.py save old   # 基线输出 -> /tmp/v18_old_*.npy
  (git stash pop 恢复后)
  python3 benchmarks/bench_final_v18.py compare    # IoU + 速度报告
"""
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, "/workspace/lineart-precise")
from lineart import to_lineart, LineArtParams


def synth_4096():
    rng = np.random.default_rng(3)
    img = np.full((4096, 4096, 3), 235, dtype=np.uint8)
    cv2.circle(img, (2048, 1536), 1000, (60, 60, 60), 12)
    cv2.rectangle(img, (512, 512), (3584, 2560), (90, 90, 90), 6)
    cv2.line(img, (600, 2700), (3600, 2700), (80, 80, 80), 3)
    noise = rng.normal(0, 6, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def samples_4096():
    out = [('synth4096', synth_4096())]
    for name in ['blueprint', 'cad_grid_bg', 'cad_thick_lines']:
        img = cv2.imread(f'/workspace/lineart-precise/demo/samples/{name}.png')
        h, w = img.shape[:2]
        s = 4096 / max(h, w)
        out.append((name, cv2.resize(img, (round(w * s), round(h * s)),
                                     interpolation=cv2.INTER_CUBIC)))
    return out


def run_save(tag):
    for name, img in samples_4096():
        t0 = time.perf_counter()
        out = to_lineart(img, LineArtParams())
        dt = time.perf_counter() - t0
        np.save(f"/tmp/v18_{tag}_{name}.npy", (out < 128))
        print(f"[{tag}] {name:16s} {dt:.1f}s ink={(out<128).sum()}")
        sys.stdout.flush()


def run_compare():
    for name, _ in samples_4096():
        try:
            a = np.load(f'/tmp/v18_old_{name}.npy')
            b = np.load(f'/tmp/v18_new_{name}.npy')
        except FileNotFoundError:
            print(f"{name}: 缺少 npy，跳过")
            continue
        iou = (a & b).sum() / max((a | b).sum(), 1)
        tol = 3
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*tol+1, 2*tol+1))
        iou_tol = (cv2.dilate(a.astype(np.uint8), k) & b).sum() / max(b.sum(), 1)
        d_new = b & ~a   # 新版独有
        d_old = a & ~b   # 旧版独有
        print(f"{name:16s} IoU={iou:.4f} tolIoU={iou_tol:.4f} "
              f"新版独有={d_new.sum():8d} 旧版独有={d_old.sum():8d}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'compare'
    if mode == 'save':
        run_save(sys.argv[2] if len(sys.argv) > 2 else 'new')
    else:
        run_compare()
