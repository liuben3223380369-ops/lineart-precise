"""公共工具函数。"""
import math

import cv2
import numpy as np


# ─── 数值常量 ────────────────────────────────────────────────────

EPS = 1e-6


# ─── 数组工具 ────────────────────────────────────────────────────

def clip_f32(arr, lo=0.0, hi=1.0):
    """裁剪 float32 数组到 [lo, hi]。"""
    return np.clip(arr, lo, hi).astype(np.float32)


def to_f32(arr):
    """转换为 float32。"""
    return np.asarray(arr, dtype=np.float32)


def to_u8(arr):
    """转换为 uint8 [0,255]。"""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


# ─── 双线性采样 ──────────────────────────────────────────────────

def bilinear_sample(img, ys, xs):
    """双线性采样：img 单通道 float32，ys/xs 为坐标数组（与 img 同形或标量）。

    越界采样返回 0（线稿背景为白=1，但亚像素精化用梯度幅值场，
    越界视为 0 避免边界伪影；与 refine 中 fit_field 边界语义一致）。"""
    h, w = img.shape
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    y1 = y0 + 1
    x1 = x0 + 1
    # 4 个角点全部在界内才有效（任一越界返回 0）
    inb = (y0 >= 0) & (y0 < h) & (x0 >= 0) & (x0 < w) & \
          (y1 >= 0) & (y1 < h) & (x1 >= 0) & (x1 < w)
    return (img[y0.clip(0, h - 1), x0.clip(0, w - 1)] * (1 - fy) * (1 - fx) +
            img[y1.clip(0, h - 1), x0.clip(0, w - 1)] * fy * (1 - fx) +
            img[y0.clip(0, h - 1), x1.clip(0, w - 1)] * (1 - fy) * fx +
            img[y1.clip(0, h - 1), x1.clip(0, w - 1)] * fy * fx) * inb


def bilinear_sample_field(img, ys, xs):
    """整场双线性采样：ys/xs 为与 img 同形的 float32 数组。"""
    return bilinear_sample(img, ys, xs)


# ─── 双线性 splat（渲染） ────────────────────────────────────────

def bilinear_splat(canvas, ys, xs, values):
    """向量化双线性 splat：把 values 按双线性权重累加到 canvas 的 (ys,xs) 位置。

    canvas: [H,W] float32，ys/xs/values: 等长数组。
    越界权重（坐标超出 [0, H-1]×[0, W-1]）被丢弃，避免把边缘线
    拉向画布中心（旧版 np.clip 会把越界权重累加到边界像素）。
    4 个角点全部在界内才累加（任一越界整个 splat 丢弃）。
    性能：4 角索引/权重拼接后单次 np.add.at（ufunc.at 每次调用
    有固定分发开销，4→1 次实测 180K 点 167ms→48ms）。
    """
    h, w = canvas.shape
    if len(ys) == 0:
        return
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    fy = ys - y0
    fx = xs - x0
    y1 = y0 + 1
    x1 = x0 + 1

    inb = (y0 >= 0) & (y0 < h) & (x0 >= 0) & (x0 < w) & \
          (y1 >= 0) & (y1 < h) & (x1 >= 0) & (x1 < w)
    v = values * inb  # 越界点 v=0 → 4 角权重全 0（clip 索引安全落界）
    # 4 角拼接：单次 add.at
    rows = np.concatenate((y0, y0, y1, y1)).clip(0, h - 1)
    cols = np.concatenate((x0, x1, x0, x1)).clip(0, w - 1)
    wgt = np.concatenate((v * (1 - fy) * (1 - fx), v * (1 - fy) * fx,
                          v * fy * (1 - fx), v * fy * fx))
    np.add.at(canvas.ravel(), rows * w + cols, wgt)


# ─── 高斯核生成 ──────────────────────────────────────────────────

def gaussian_kernel_1d(sigma, radius=None):
    """生成 1D 高斯核。"""
    if radius is None:
        radius = max(1, int(round(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(xs ** 2) / (2.0 * sigma * sigma + EPS))
    return g / g.sum()


# ─── 形态学工具 ──────────────────────────────────────────────────

def morph_open(mask, ksize=3):
    """形态学开运算（去小突起）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel) > 0)


def morph_close(mask, ksize=3):
    """形态学闭运算（弥合小断点）。"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    return (cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0)


# ─── 连通域工具 ──────────────────────────────────────────────────

def remove_small_components(mask, min_area):
    """剔除面积过小的连通域。

    查表法：逐标签循环做全图比较是 O(标签数×N)（4096 图 45k 标签时 537ms），
    改为标签 LUT + 一次花式索引 O(N)。
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.ones(num, dtype=bool)
    keep[0] = False  # 背景标签恒为 False
    if num > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


# ─── 图像缩放 ────────────────────────────────────────────────────

MAX_SIDE = 4096


def limit_size(img, max_side=MAX_SIDE):
    """限制图像长边。"""
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / float(side)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return img, scale
