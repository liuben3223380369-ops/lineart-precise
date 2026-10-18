"""Sauvola 局部自适应二值化语义层。

适用场景（全局阈值失效的情况）：
- 手机拍照的手绘草图：纸张光照不均（阴影、倾角渐变）
- 扫描文档/手稿：折痕、污渍、装订阴影
- 铅笔线本身是弱渐变灰，全局阈值要么丢弱线、要么把阴影判成线

Sauvola (2000) 公式：
    T(x,y) = m(x,y) * (1 + k * (s(x,y)/R - 1))
    m = 局部均值, s = 局部标准差, R = 动态范围的一半（归一化灰度取 0.5）
    k ∈ [0.1, 0.34]，越大越严格（阈值越低）

实现用 cv2.boxFilter（C++ 滑窗增量求和）：
    m = boxFilter(g)，E[g^2] = boxFilter(g*g)，var = E[g^2] - m^2
    每像素 O(1) 且全程 float32，无 float64 中间数组与积分图大数差分
    （4096 图 351ms → ~110ms）。boxFilter 的 BORDER_REPLICATE 与
    copyMakeBorder(REPLICATE) + 积分图窗口的边界语义逐像素一致。
"""
import cv2
import numpy as np


def sauvola_threshold(gray, window=None, k=0.2, r_dynamic=0.5):
    """计算 Sauvola 自适应阈值图。

    gray: [0,1] float32 灰度图
    window: 窗口边长（奇数）。None 时按图像尺寸自适应（min(h,w)/20，夹在 15~75）
    返回 (threshold, mean, std)，均为 float32，与 gray 同形
    """
    h, w = gray.shape
    if window is None:
        window = int(np.clip(round(min(h, w) / 20.0), 15, 75))
    window = max(3, int(window) | 1)  # 保证奇数

    ksize = (window, window)
    m = cv2.boxFilter(gray, -1, ksize, normalize=True,
                      borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(gray * gray, -1, ksize, normalize=True,
                            borderType=cv2.BORDER_REPLICATE)
    var = mean_sq
    var -= m * m
    np.maximum(var, 0.0, out=var)
    s = var
    np.sqrt(s, out=s)

    # t = m * (1 + k*(s/R - 1)) = m * ((1-k) + s*k/R)
    t = s * (k / r_dynamic)
    t += (1.0 - k)
    t *= m
    return t, m, s


def sensitivity_to_k(sensitivity):
    """line_sensitivity [0,1] → Sauvola k。sensitivity 越高（要更多线）k 越小。"""
    sens = float(np.clip(sensitivity, 0.0, 1.0))
    return 0.34 - 0.20 * sens


def adaptive_semantic(gray, window=None, sensitivity=0.5, softness=0.5):
    """自适应阈值语义层（与 fdog_semantic / multiscale_xdog 同接口）。

    返回 [0,1] float32 ink 图：1 = 线。软过渡避免硬边界影响梯度验证层。

    性能：>2048 大图时阈值场与软过渡 ink 都在 0.75 尺度计算，仅对最终
    ink 做一次上采样（阈值/带宽场本质低频；相较在 t/s 各自上采样后再算
    ink，省两次全图 resize 与 40% 的逐像素运算，软过渡带翻转 <0.1%，
    线条位置由梯度定位层保证不受影响）。
    """
    k = sensitivity_to_k(sensitivity)
    h, w = gray.shape
    large = max(h, w) > 2048
    if large:
        nw, nh = int(round(w * 0.75)), int(round(h * 0.75))
        small = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
        t, _, s = sauvola_threshold(small, window=window, k=k)
        gray = small
    else:
        t, m, s = sauvola_threshold(gray, window=window, k=k)

    # 软过渡带宽：与局部对比度挂钩（对比度高的地方过渡更宽）
    band = np.maximum(s * 0.6, 0.015)
    band = band * (0.5 + softness) + 0.01

    ink = (t - gray) / (2.0 * band) + 0.5
    np.clip(ink, 0.0, 1.0, out=ink)
    if large:
        ink = cv2.resize(ink, (w, h), interpolation=cv2.INTER_LINEAR)
    return ink.astype(np.float32, copy=False)
