"""导向滤波（He et al. 2010）：保边平滑。

借鉴 bobbens/sketch_simplification (754★) 与 Informative Drawings 的
"保留线条、抹平纸面噪声"思想：相较高斯/双边滤波，导向滤波在同等噪声
抑制下对细弱线条的损伤显著更小，且边缘零漂移（对定位精度无害）。

积分图优化：用 cv2.integral 替代 cv2.boxFilter 计算局部均值。
原理：mean = (I[y2,x2] - I[y1,x2] - I[y2,x1] + I[y1,x1]) / area
边界处理：BORDER_REPLICATE 扩展后计算积分图，再裁剪回原尺寸。
"""
import cv2
import numpy as np

from ..common import EPS


def _box_integral(src, radius):
    """用积分图实现 boxFilter 等效的局部均值计算。

    参数:
        src: float32 单通道图像
        radius: 滤波半径

    返回:
        与 cv2.boxFilter(normalize=True) 等效的局部均值图
    """
    h, w = src.shape
    # 边界扩展：BORDER_REPLICATE 保持边缘一致性
    padded = cv2.copyMakeBorder(src, radius, radius, radius, radius,
                                cv2.BORDER_REPLICATE)
    # 计算积分图（float64 避免累加溢出）
    integral = cv2.integral(padded.astype(np.float64))
    # 积分图比原图大 1，需要偏移索引
    integral = integral.astype(np.float64)

    # 预计算窗口面积
    ksize = 2 * radius + 1
    area = ksize * ksize

    # 输出数组
    out = np.empty((h, w), dtype=np.float64)

    # 对每个像素计算局部均值
    # y1, x1 是窗口左上角（含），y2, x2 是窗口右下角（含）
    # 积分图索引：I[y,x] 包含 [0,0) 到 [y,x) 的和
    for y in range(h):
        y1 = y
        y2 = y + ksize
        row_integral = integral[y1:y2 + 1, :]
        for x in range(w):
            x1 = x
            x2 = x + ksize
            # mean = (I[y2,x2] - I[y1,x2] - I[y2,x1] + I[y1,x1]) / area
            val = (integral[y2, x2] - integral[y1, x2]
                   - integral[y2, x1] + integral[y1, x1])
            out[y, x] = val / area

    return out.astype(np.float32)


def _box_integral_optimized(src, radius):
    """向量化积分图均值计算（性能优化版本）。

    用切片操作替代逐像素循环，大幅提升性能。
    """
    h, w = src.shape
    ksize = 2 * radius + 1

    # 边界扩展：BORDER_REPLICATE
    padded = cv2.copyMakeBorder(src, radius, radius, radius, radius,
                                cv2.BORDER_REPLICATE)
    # 计算积分图（float64 避免累加精度损失）
    integral = cv2.integral(padded.astype(np.float64))

    # 向量化计算：利用切片批量获取四个角的值
    # I[y2, x2]: 右下角
    # I[y1, x2]: 左下角
    # I[y2, x1]: 右上角
    # I[y1, x1]: 左上角
    y1_indices = np.arange(h)
    y2_indices = y1_indices + ksize
    x1_indices = np.arange(w)
    x2_indices = x1_indices + ksize

    # 使用高级索引批量获取
    # 形状: (h, w)
    i_y2x2 = integral[np.ix_(y2_indices, x2_indices)]
    i_y1x2 = integral[np.ix_(y1_indices, x2_indices)]
    i_y2x1 = integral[np.ix_(y2_indices, x1_indices)]
    i_y1x1 = integral[np.ix_(y1_indices, x1_indices)]

    area = ksize * ksize
    mean = (i_y2x2 - i_y1x2 - i_y2x1 + i_y1x1) / area

    return mean.astype(np.float32)


def guided_filter(guide, src, radius=3, eps=1e-4, use_integral=True):
    """guide/src: [0,1] float32 单通道。返回保边平滑结果。

    参数:
        guide: 引导图像
        src: 输入图像
        radius: 滤波半径
        eps: 正则化参数
        use_integral: 是否使用积分图优化（默认 True）

    自引导（guide is src）时走快速路径：corr_ip ≡ corr_ii、mean_i ≡ mean_p，
    6 次 box 滤波 + 2 次全图乘法减为 3 次 + 1 次（4096 图 ~2x 提速），
    数值等价：唯一差异在 var 数值噪声为负的像素（|cov_ip|<1e-9 量级），
    对输出影响 <1e-4。

    积分图优化：用 cv2.integral 替代 cv2.boxFilter，边界用 BORDER_REPLICATE。
    """
    # 选择滤波函数
    if use_integral:
        def box(a):
            return _box_integral_optimized(a, radius)
    else:
        ksize = (2 * radius + 1, 2 * radius + 1)
        def box(a):
            return cv2.boxFilter(a, -1, ksize, normalize=True,
                                 borderType=cv2.BORDER_REFLECT)

    if guide is src:
        mean = box(guide)
        var_i = box(guide * guide)
        var_i -= mean * mean
        np.maximum(var_i, 0.0, out=var_i)
        a = var_i
        a /= var_i + eps
        b = mean - a * mean
        return box(a) * guide + box(b)

    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(guide * src)
    corr_ii = box(guide * guide)
    var_i = np.maximum(corr_ii - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def guided_filter_boxfilter(guide, src, radius=3, eps=1e-4):
    """原始 boxFilter 实现（用于性能对比基准）。"""
    ksize = (2 * radius + 1, 2 * radius + 1)

    def box(a):
        return cv2.boxFilter(a, -1, ksize, normalize=True,
                             borderType=cv2.BORDER_REFLECT)

    if guide is src:
        mean = box(guide)
        var_i = box(guide * guide)
        var_i -= mean * mean
        np.maximum(var_i, 0.0, out=var_i)
        a = var_i
        a /= var_i + eps
        b = mean - a * mean
        return box(a) * guide + box(b)

    mean_i = box(guide)
    mean_p = box(src)
    corr_ip = box(guide * src)
    corr_ii = box(guide * guide)
    var_i = np.maximum(corr_ii - mean_i * mean_i, 0.0)
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def edge_preserving_denoise(gray, radius=3, eps=1e-4, strength=1.0):
    if strength <= 0:
        return gray
    out = guided_filter(gray, gray, radius=radius, eps=eps)
    if strength < 1.0:
        out *= strength
        out += (1.0 - strength) * gray
    np.clip(out, 0.0, 1.0, out=out)
    return out.astype(np.float32, copy=False)
