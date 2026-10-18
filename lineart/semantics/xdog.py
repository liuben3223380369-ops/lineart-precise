"""多尺度软阈值 XDoG：产生连续灰度线稿响应，暗=线。"""
import cv2
import numpy as np


def xdog_response(gray, sigma, p=25.0, eps=0.012, phi=10.0, k=1.6):
    """单尺度 XDoG。gray: [0,1] float32，返回 [0,1] float32（1=白 0=黑线）。"""
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    u = (1.0 + p) * g1 - p * g2
    soft = 1.0 + np.tanh(phi * (u - eps))
    soft = np.where(u >= eps, 1.0, soft)
    return np.clip(soft, 0.0, 1.0)


def multiscale_xdog(gray, sigmas=(0.8, 1.8), p=25.0, eps=0.012, phi=10.0):
    """多尺度 XDoG 融合。细尺度保细节，粗尺度保主轮廓，取逐点最强墨迹。"""
    inks = []
    for s in sigmas:
        resp = xdog_response(gray, s, p=p, eps=eps, phi=phi)
        inks.append(1.0 - resp)
    stack = np.stack(inks, axis=0)
    ink = stack.max(axis=0)
    return np.clip(ink, 0.0, 1.0).astype(np.float32)
