"""高精度白底/黑底多色线转换主管线（编排层）。

语义-定位解耦架构（第 9 轮重组后按管线阶段分包）：
- preprocess/   预处理层：底色/光照/去噪/文字/冲击锐化
- semantics/    语义层（哪里有线，可插拔）：FDoG / XDoG / Sauvola / 神经先验
- localization/ 定位层（线在哪里，精度保证）：梯度 / NMS / 亚像素精化 / 渲染

CAD 增强：
- 自适应底色检测与反转（黑/白/灰/混合/蓝图）
- 多色线分离与颜色保留渲染
- 网格/点阵背景去除
- 文字区域检测与可选去除
- 粗线条中心线提取（Zhang-Suen 细化）

场景自适应（scene="auto"）：
- 自动识别 像素艺术 / 蓝图 / 拍照手绘 / 扫描文档 / 数字线稿 / 照片
- 按场景覆盖参数与预处理（像素图最近邻预放大、手绘走 adaptive 语义等）
"""
import cv2
import numpy as np

from .color import extract_color_layers, render_color_lineart, to_color_lineart
from .common import remove_small_components
from .io import coerce_bgr, limit_size, to_gray
from .localization.edges import (
    bridge_gaps,
    extract_centerline,
    hysteresis,
    medial_skeleton,
    nonmax_suppression,
    remove_grid_bg,
    verify_with_gradient,
)
from .localization.gradient import gradient_farid, gradient_sobel
from .localization.render import refine_and_render
from .params import LineArtParams
from .preprocess.background import normalize_background
from .preprocess.denoise import edge_preserving_denoise
from .preprocess.illumination import normalize_illumination
from .preprocess.shock import shock_sharpen as _shock_sharpen
from .preprocess.text import detect_text_mser, inpaint_text_regions
from .scenes import detect_scene, scene_info_for
from .semantics import adaptive_semantic, fdog_semantic, multiscale_xdog


def _denoise_gray(gray, strength):
    if strength <= 0:
        return gray
    tmp = np.multiply(gray, 255.0)
    np.clip(tmp, 0.0, 255.0, out=tmp)
    u8 = tmp.astype(np.uint8)
    sigma = 0.4 + 0.9 * min(strength, 1.0)
    u8 = cv2.GaussianBlur(u8, (0, 0), sigma)
    d = 15.0 + 75.0 * min(strength, 1.0)
    u8 = cv2.bilateralFilter(u8, 5, d, d)
    out = u8.astype(np.float32)
    out *= 1.0 / 255.0
    return out


def _sigmas_from_detail(detail):
    d = float(np.clip(detail, 0.0, 1.0))
    fine = 0.7 + (1.0 - d) * 0.9
    coarse = 1.5 + (1.0 - d) * 1.8
    return (fine, coarse)


def _eps_from_sensitivity(sensitivity):
    return 0.030 - 0.024 * float(np.clip(sensitivity, 0.0, 1.0))


def _build_mask(strength, mag, gx, gy, p, nms_override=None):
    """通用掩码构建：滞回+NMS+骨架+连通域清理。"""
    if nms_override is None:
        # 梯度幅值最低门槛：剔除双边滤波/亮区渐变产生的虚假弱响应
        # 用峰值比例：只保留 >15% 最大梯度的 NMS 点
        mag_min = float(mag.max()) * 0.15
        nms = nonmax_suppression(mag, gx, gy, mag_min=mag_min)
    else:
        nms = nms_override
    mask = hysteresis(strength)
    mask = mask & nms
    mask = bridge_gaps(mask, closing_size=3)
    mask = mask & nms
    mask = medial_skeleton(mask)
    mask = remove_small_components(mask, min_area=p.min_area)
    if p.extract_centerline and mask.any():
        skel = extract_centerline(mask)
        if skel.any():
            mask = skel
    return mask


def _render_grayscale(mag, mask, gx, gy, strength, thickness,
                      method="gaussian", render_ss=1, consensus=True,
                      merge_edges=False, merge_gap=6, dark_field=None):
    dark_map = np.multiply(strength, 1.6)
    np.clip(dark_map, 0.0, 1.0, out=dark_map)
    canvas = refine_and_render(mag, mask, gx, gy,
                                        darkness=dark_map,
                                        thickness=thickness,
                                        method=method,
                                        render_ss=render_ss,
                                        consensus=consensus,
                                        merge_edges=merge_edges,
                                        merge_gap=merge_gap,
                                        dark_field=dark_field)
    np.subtract(1.0, canvas, out=canvas)
    canvas *= 255.0
    np.clip(canvas, 0.0, 255.0, out=canvas)
    return canvas.astype(np.uint8)


def _render_binary(canvas_gray):
    return np.where(canvas_gray >= 128, 255, 0).astype(np.uint8)


def _apply_scene(img_bgr, p):
    """场景自适应：检测（或采用指定）场景，返回（可能预处理的）图像与覆盖参数后的副本。

    像素艺术在原书上做最近邻预放大（产生可亚像素精化的过渡带）。
    """
    if p.scene == "off":
        return img_bgr, p
    if p.scene == "auto":
        info = detect_scene(img_bgr)
    else:
        info = scene_info_for(p.scene)

    overrides = {k: v for k, v in info.params.items() if v is not None}
    new_p = p.__class__(**{**p.__dict__, **overrides}) if overrides else p

    up = info.pre.get("upscale")
    if up and up > 1.0:
        img = cv2.resize(img_bgr, None, fx=up, fy=up,
                         interpolation=cv2.INTER_NEAREST)
        img, _ = limit_size(img, new_p.max_side)
        return img, new_p
    return img_bgr, new_p


def to_lineart(img_bgr, params=None, semantic_map=None):
    """输入 BGR/RGBA 图像，输出线稿图。

    输出格式由 params.output 决定：
    - "grayscale": 白底黑线 uint8（255=白）
    - "binary": 纯黑白 uint8
    - "color": 白底彩色线稿 uint8 BGR（保留 CAD 线条颜色）
    - "layered": 分层彩色线稿 list of (mask, color_bgr)

    输出空间尺寸与输入一致（scene=pixel 的内部预放大会在输出前缩回）。
    semantic_map: 可选外部神经语义先验（[0,1] float，1=线）。
    """
    p = params or LineArtParams()
    img_bgr = coerce_bgr(img_bgr)  # 标准化 dtype/通道（宽容灰度/float/RGBA/空图防御）
    orig_hw = (img_bgr.shape[0], img_bgr.shape[1])
    img_bgr, p = _apply_scene(img_bgr, p)
    scene_upscaled = (img_bgr.shape[0], img_bgr.shape[1]) != orig_hw
    img, _ = limit_size(img_bgr, p.max_side)
    text_mask = None
    if p.remove_text:
        gray_raw = to_gray(img)
        text_mask = detect_text_mser((gray_raw * 255).astype(np.uint8))

    if p.auto_invert:
        img, inverted, mode = normalize_background(img)
    else:
        inverted = False
        mode = "white"

    gray = to_gray(img)

    if p.remove_grid:
        gray, _ = remove_grid_bg(
            gray, period_range=(p.grid_period_min, 20))

    gray = normalize_illumination(gray, strength=p.illumination)
    gray = edge_preserving_denoise(gray, radius=p.smooth_radius,
                                   eps=1e-4, strength=p.denoise)
    gray = _denoise_gray(gray, p.denoise * 0.5)

    if text_mask is not None and text_mask.sum() > 0:
        img = inpaint_text_regions(img, text_mask)
        gray = to_gray(img)

    # 检测/定位数据流分离（sharpen>0 时）：
    # - 检测图 gray_det：冲击锐化后过渡带收窄成准阶跃 → 梯度峰更锐，
    #   语义提取更稳、验证强度更高（输出线更黑）；
    # - 拟合图 gray_fit：保留原始连续过渡带 → 高斯拟合的亚像素相位
    #   信息完整（锐化会把中间灰度推向 0/1，位置量化到整数网格）。
    gray_fit = gray
    gray_det = _shock_sharpen(gray, strength=p.sharpen)

    if semantic_map is not None:
        h, w = gray.shape
        sm = semantic_map
        if sm.shape != (h, w):
            sm = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)
        ink = np.clip(sm, 0.0, 1.0).astype(np.float32)
    elif p.semantic == "fdog":
        ink = fdog_semantic(gray_det, coherence=p.coherence,
                            detail=p.detail,
                            sensitivity=p.line_sensitivity)
    elif p.semantic == "adaptive":
        ink = adaptive_semantic(gray_det, sensitivity=p.line_sensitivity)
    else:
        ink = multiscale_xdog(gray_det,
                              sigmas=_sigmas_from_detail(p.detail),
                              eps=_eps_from_sensitivity(p.line_sensitivity))

    def _grad(g):
        if p.gradient_method == "farid":
            return gradient_farid(g)
        return gradient_sobel(g)

    # 定位梯度（拟合场）取原始图；检测梯度取锐化图（sharpen=0 时共享）
    mag, gx, gy = _grad(gray_fit)
    if gray_det is gray_fit:
        mag_d, gx_d, gy_d = mag, gx, gy
    else:
        mag_d, gx_d, gy_d = _grad(gray_det)
    strength = verify_with_gradient(ink, mag_d, gamma=p.gradient_gamma)
    mask = _build_mask(strength, mag_d, gx_d, gy_d, p)

    if inverted:
        pass  # normalize_background 已反转图像，mask 已是白底黑线方向，无需再反转

    if p.output == "layered":
        layers, _ = extract_color_layers(img, n_colors=p.n_colors)
        return layers

    if p.output == "color":
        out = to_color_lineart(img, n_colors=p.n_colors, thickness=p.thickness)
        if p.mode == "binary":
            gray_out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
            out = np.where(gray_out >= 128, 255, 0).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        return out

    # pixel 场景的最近邻预放大已产生整数过渡带，SSR 的软墨会破坏
    # block-max 缩回的硬边保持，故预放大管线禁用超采样渲染。
    render_ss = 1 if scene_upscaled else p.render_ss
    canvas_gray = _render_grayscale(mag, mask, gx, gy, strength,
                                    p.thickness, method=p.subpixel_method,
                                    render_ss=render_ss,
                                    consensus=p.consensus,
                                    merge_edges=p.merge_edges,
                                    merge_gap=p.merge_gap,
                                    dark_field=gray_fit)
    out = _render_binary(canvas_gray) if p.mode == "binary" else canvas_gray
    return _restore_size(out, orig_hw, scene_upscaled,
                         binary=(p.mode == "binary"))


def _restore_size(out, orig_hw, upscaled, binary):
    """像素艺术预放大后缩回原尺寸，保持"输出尺寸=输入尺寸"的 API 契约。

    缩回用 block-max（4x4 块取最暗）：4px 线缩回 1px 恢复完整深度
    （INTER_AREA 平均会把软 splat 线稀释到 ~0.45 墨迹）。
    尺寸非整数倍时退回 INTER_AREA。
    binary 模式缩回产生的灰阶重新二值化。
    """
    if not upscaled or out.shape[:2] == orig_hw:
        return out
    if isinstance(out, list):
        return out
    oh, ow = orig_hw
    h, w = out.shape[:2]
    if h % oh == 0 and w % ow == 0 and (h // oh) == (w // ow):
        f = h // oh
        if f > 1:
            if out.ndim == 2:
                blocked = out.reshape(oh, f, ow, f).min(axis=(1, 3))
            else:
                c = out.shape[2]
                blocked = out.reshape(oh, f, ow, f, c).min(axis=(1, 3))
            out = blocked
    else:
        out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
    if binary:
        out = np.where(out >= 128, 255, 0).astype(np.uint8)
    return out
