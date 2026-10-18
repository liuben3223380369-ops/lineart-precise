"""v1.7.0 第三版边缘合并验证：粗线双边缘 → 线心。

测试图：256x256 白底，2px 硬边矩形 + 2px 水平线（灰度 40），
GT 线心：矩形边缘在 x/y=60.5/196.5，水平线在 y=128。
指标：虚假墨（GT 外墨像素）、丢线（GT 内空白）、y100/x128 剖面。
"""
import cv2
import numpy as np

from lineart import to_lineart, LineArtParams

H = W = 256
GT_LINE = 40.0


def make(noise: bool) -> np.ndarray:
    clean = np.full((H, W), 255.0, np.float32)
    cv2.rectangle(clean, (60, 60), (196, 196), GT_LINE, 2)
    cv2.line(clean, (20, 128), (236, 128), GT_LINE, 2)
    if not noise:
        return clean
    rng = np.random.default_rng(7)
    blot = cv2.GaussianBlur(rng.normal(0, 12, (H, W)).astype(np.float32), (31, 31), 10)
    grain = rng.normal(0, 8, (H, W)).astype(np.float32)
    return clean + blot * 10.0 + grain


def gt_mask() -> np.ndarray:
    """GT = 原始图真实墨位置（实测：边 59-61/195-197，水平线 127-129）。"""
    gt = np.zeros((H, W), bool)
    gt[59:62, 59:198] = True         # 上边
    gt[195:198, 59:198] = True       # 下边
    gt[62:195, 59:62] = True         # 左边
    gt[62:195, 195:198] = True       # 右边
    gt[127:130, 20:237] = True       # 水平线
    return gt


def metrics(out: np.ndarray, gt: np.ndarray):
    ink = out < 128  # 墨=暗（灰度抗锯齿或二值均适用）
    fp = int(np.count_nonzero(ink & ~gt))
    gt_ink = int(np.count_nonzero(gt))
    miss = int(np.count_nonzero(~ink & gt))
    return fp, miss, gt_ink


def main():
    gt = gt_mask()
    for label, noise in (("干净", False), ("噪声", True)):
        bgr = cv2.cvtColor(np.clip(make(noise), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        for me in (False, True):
            out = to_lineart(bgr, LineArtParams(output="binary", mode="binary",
                                                merge_edges=me, scene="off"))
            fp, miss, gt_ink = metrics(out, gt)
            y100 = np.where(out[100] < 128)[0].tolist()
            x128 = np.where(out[:, 128] < 128)[0].tolist()
            print(f"[{label} merge={me}] 虚假墨={fp}px 丢线={miss}/{gt_ink} ({miss/gt_ink:.0%})")
            print(f"    y=100 行墨: {y100}")
            print(f"    x=128 列墨: {x128}")


if __name__ == "__main__":
    main()
