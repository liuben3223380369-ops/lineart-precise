"""高精度白底/黑底多色线转换主管线（编排层）。

批量 API（v1.9.0）：
- to_lineart_batch：并行处理图像序列，按输入顺序产出结果
- to_lineart_batch_simple：简化版批量 API

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

import concurrent.futures
import os
from collections.abc import Iterable, Iterator
from typing import Any, Generator, Optional, Tuple, Union

try:
    from tqdm import tqdm as _tqdm

    _HAS_TQDM = True
except ImportError:
    _HAS_TQDM = False

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
from .localization import bridge_broken_lines
from .localization.gradient import gradient_farid, gradient_sobel
from .localization.render import refine_and_render
from .localization.render_shape import build_shape_mask, shape_faithful_render
from dataclasses import fields

from .params import LineArtParams
from .preprocess.background import normalize_background
from .preprocess.denoise import edge_preserving_denoise
from .preprocess.illumination import normalize_illumination
from .preprocess.shock import shock_sharpen as _shock_sharpen
from .preprocess.text import detect_text_mser, inpaint_text_regions
from .scenes import detect_scene, scene_info_for
from .semantics import adaptive_semantic, fdog_semantic, multiscale_xdog
from .semantics.neural import (dexined_semantic, ensemble_semantic,
                               neural_postprocess, neural_semantic)


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
    """通用掩码构建：滞回+NMS+骨架+连通域清理+断点桥接。"""
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
    # 断点桥接：距离约束的端点近邻连接（方向信息不再需要）
    if p.merge_edges and p.bridge_gap > 1:
        mask = bridge_broken_lines(mask, gx, gy, None, None,
                                   max_gap=p.bridge_gap, min_area=p.min_area)
    return mask


def _to_white_black_u8(ink01):
    """墨迹 [0,1]（1=全墨）→ 白底黑线 uint8（255=白）。"""
    canvas = np.subtract(1.0, ink01)
    canvas *= 255.0
    np.clip(canvas, 0.0, 255.0, out=canvas)
    return canvas.astype(np.uint8)


def _render_grayscale(mag, mask, gx, gy, strength, thickness,
                      method="gaussian", render_ss=1, consensus=True,
                      merge_edges=False, merge_gap=20, dark_field=None,
                      adaptive_width=True, uniform_width=False):
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
                                        dark_field=dark_field,
                                        adaptive_width=adaptive_width,
                                        uniform_width=uniform_width)
    return _to_white_black_u8(canvas)


def _render_binary(canvas_gray):
    return np.where(canvas_gray >= 128, 255, 0).astype(np.uint8)


def _bridge_uniform_gaps(canvas_gray):
    """统一线宽（v1.16.1）的角部断口连接。

    粗线配对合并在拐角处覆盖不全（法向旋转致配对失败），残余
    脊线点与合并中线错位半线宽，细线渲染下显形为断口。闭运算
    不增粗线条（直线无凹缺），只连接 <=6px 断口——实测 6px 粗
    矩形框四角断口 6.8px 斜距，closing=7（半径 3 膨胀可达）恰好
    覆盖且不误并密集平行线（间距 >7px 不受影响）。
    """
    ink = canvas_gray < 128
    if not ink.any():
        return canvas_gray
    closed = bridge_gaps(ink, 7)
    fill = np.logical_and(closed, np.logical_not(ink))
    if not fill.any():
        return canvas_gray
    out = canvas_gray.copy()
    out[fill] = 38  # 连接段固定深墨（≈0.85 浓度，与 splat 线视觉一致）
    return out


def _apply_scene(img_bgr, p):
    """场景自适应：检测（或采用指定）场景，返回（可能预处理的）图像、
    覆盖参数后的副本与场景名。

    像素艺术在原书上做最近邻预放大（产生可亚像素精化的过渡带）。

    v1.13.0 覆盖语义：场景默认值只覆盖「用户未调整」的参数（当前值
    仍等于 LineArtParams dataclass 默认值的字段）。用户显式修改过的
    参数保持用户值——此前场景默认会静默覆盖用户的手动调参（照片
    弱边缘调 denoise/sensitivity 被 photo 场景的 0.55/0.5 顶掉，
    用户完全无法调优）。

    v1.21.0 返回场景名：供 dexined_scale="auto" 场景路由（清晰
    线稿类场景 2x 高分辨率推理，photo/sketch 1x）。
    """
    if p.scene == "off":
        return img_bgr, p, "off"
    if p.scene == "auto":
        info = detect_scene(img_bgr)
    else:
        info = scene_info_for(p.scene)

    defaults = {f.name: getattr(p.__class__, f.name)
                for f in fields(p.__class__)}
    overrides = {k: v for k, v in info.params.items()
                 if v is not None and getattr(p, k, None) == defaults.get(k)}
    new_p = p.__class__(**{**p.__dict__, **overrides}) if overrides else p

    up = info.pre.get("upscale")
    if up and up > 1.0:
        img = cv2.resize(img_bgr, None, fx=up, fy=up,
                         interpolation=cv2.INTER_NEAREST)
        img, _ = limit_size(img, new_p.max_side)
        return img, new_p, info.scene
    return img_bgr, new_p, info.scene


_DEXINED_SCALE_SCENES = {"lineart": 2, "blueprint": 2, "pixel": 2}


def _dexined_scale_for(p, scene):
    """dexined_scale 解析：显式 "1"/"2" 优先，"auto" 按场景路由。

    清晰硬边缘场景（lineart/blueprint/pixel）→ 2（大图定位偏差
    -43%、锐度 +95%）；弱对比软边缘场景（photo/sketch/off/未知）
    → 1（2x 感受野失配，山脊召回 -12pp）。
    """
    if p.dexined_scale in ("1", "2"):
        return int(p.dexined_scale)
    return _DEXINED_SCALE_SCENES.get(scene, 1)


def _render_canvas_gray(strength, ink, mag, gx, gy, mag_d, gx_d, gy_d,
                        gray_fit, neural_source, scene_upscaled, p):
    """渲染阶段：shape/splat 路线分流，返回 canvas_gray uint8。"""
    if p.render_mode == "shape":
        # 外形保真路线：原图线条外形为主（滞回区域掩膜，不 NMS/
        # 骨架化/边缘合并），管线检测为辅（响应=墨度软边、连通域清理）。
        # normalize_background 反转过的图 strength/mask 已是白底黑线方向。
        shape_mask = build_shape_mask(strength, gray_field=gray_fit,
                                      closing=p.shape_closing,
                                      min_area=p.min_area)
        ink = shape_faithful_render(strength, shape_mask,
                                    gray_field=gray_fit,
                                    ink_mode=p.shape_ink)
        return _to_white_black_u8(ink)
    if neural_source:
        # 神经链：位置/连通由神经图后处理决定（hysteresis 直通），
        # 不走原图 NMS——神经置信峰位与原图梯度峰位错开 1~4px，
        # 强联合会砍掉大部分召回（交集实测仅 11%）。
        mask = neural_postprocess(ink, p)
    else:
        mask = _build_mask(strength, mag_d, gx_d, gy_d, p)
    # pixel 场景的最近邻预放大已产生整数过渡带，SSR 的软墨会破坏
    # block-max 缩回的硬边保持，故预放大管线禁用超采样渲染。
    # 神经路径同理：mask 是滞回双脊线（v1.17.1 起不再骨架化，
    # 双线由渲染端 pair_edge_mask 配对合并取中点），超采样下采样
    # 会把线心稀释成浅灰（<128 墨点 1328→1046，视觉线淡边缘羽化）。
    render_ss = 1 if (scene_upscaled or neural_source) else p.render_ss
    # 神经路径渲染厚度保底 1.2px：滞回双脊线在细线处退化为单条
    # 2px 带（配对条件 kk>=2 不满足时保持原位），thickness=1.0
    # 默认下 splat 出浅灰断续线（<128 墨点 1425→1046，binary
    # 模式下线大量消失）。
    render_thickness = max(p.thickness, 1.2) if neural_source else p.thickness
    canvas_gray = _render_grayscale(mag, mask, gx, gy, strength,
                                    render_thickness, method=p.subpixel_method,
                                    render_ss=render_ss,
                                    consensus=p.consensus,
                                    merge_edges=p.merge_edges,
                                    merge_gap=p.merge_gap,
                                    dark_field=gray_fit,
                                    adaptive_width=p.adaptive_width,
                                    uniform_width=p.uniform_width)
    if p.uniform_width:
        canvas_gray = _bridge_uniform_gaps(canvas_gray)
    return canvas_gray


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
    img_bgr, p, scene_name = _apply_scene(img_bgr, p)
    scene_upscaled = (img_bgr.shape[0], img_bgr.shape[1]) != orig_hw
    img, _ = limit_size(img_bgr, p.max_side)
    # 神经语义在预处理链之前推理：informative-drawings 的训练分布是
    # 自然照片，管线的光照归一化/保边去噪会抹掉弱结构边缘（实测弱
    # 对比山脊带响应 p90 0.34→0.004），背景反相判定在照片场景亦会误判。
    # 神经源（semantic="neural" 或外部 semantic_map）整链跳过针对
    # 传统检测链的预处理：光照归一/保边去噪会软化边缘，SSR 的 FWHM
    # 宽度展开随之把线宽膨胀（实测 2px 线渲染成 6px，两路径 IoU
    # 0.614→0.99+），背景反相判定在照片场景亦会误判；神经链的位置/
    # 宽度由神经图后处理决定，与原图过渡带无关（v1.18.0 统一）。
    neural_source = (semantic_map is not None
                     or p.semantic in ("neural", "dexined", "ensemble"))
    neural_ink = None
    if p.semantic == "neural":
        neural_ink = neural_semantic(img, p.neural_model)
    elif p.semantic == "dexined":
        neural_ink = dexined_semantic(img, p.neural_model,
                                      scale=_dexined_scale_for(p, scene_name))
    elif p.semantic == "ensemble":
        neural_ink = ensemble_semantic(img, p.neural_model,
                                       scale=_dexined_scale_for(p, scene_name))
    text_mask = None
    if p.remove_text:
        gray_raw = to_gray(img)
        text_mask = detect_text_mser((gray_raw * 255).astype(np.uint8))

    if p.auto_invert and not neural_source:
        img, _, _ = normalize_background(img)
        # 反转后的图 strength/mask 已是白底黑线方向，渲染无需再反转

    gray = to_gray(img)

    if not neural_source:
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
    elif p.semantic in ("neural", "dexined", "ensemble"):
        ink = neural_ink
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
    # 神经语义置信度经墨度补偿后作墨度场。verify_with_gradient 的
    # 原图梯度增益门控会把纹理弱响应放大成噪线（实测草地场景
    # 3.98%→7.38%），故神经路径禁用；
    # ×2.2 补偿：神经弱响应（≈0.34）在默认 thickness=1.0 下渲染成
    # 浅灰线（>128，binary 阈值下不可见），补偿后覆盖 61.9%→70.8%。
    if neural_source:
        strength = np.clip(ink * 2.2, 0.0, 1.0).astype(np.float32)
    else:
        strength = verify_with_gradient(ink, mag_d, gamma=p.gradient_gamma)

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

    canvas_gray = _render_canvas_gray(strength, ink, mag, gx, gy,
                                      mag_d, gx_d, gy_d, gray_fit,
                                      neural_source, scene_upscaled, p)
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


def _default_workers():
    """默认 CPU 核数，最少 1。"""
    n = os.cpu_count() or 1
    return max(1, n)


def to_lineart_batch(
    images: Union[Iterable, Iterator],
    params: Optional[Any] = None,
    n_workers: Optional[int] = None,
    progress: bool = False,
    semantic_map: Optional[Any] = None,
) -> Generator[Tuple[int, Any, Optional[Exception]], None, None]:
    """批量并行转换线稿，按输入顺序产出结果。

    参数：
        images: 可迭代序列（list/generator/迭代器），每个元素为 BGR/RGBA 图像
        params: LineArtParams 实例，None 时使用默认值
        n_workers: 并行线程数，None 时自动检测 CPU 核数
        progress: 是否显示 tqdm 进度条（需要安装 tqdm）
        semantic_map: 可选外部神经语义先验，每张图独立传入（不支持批量）

    产出：
        (index, result, exception) 元组：
        - index: 输入顺序索引（0 起），用于排序与对齐
        - result: 成功时的线稿图，失败时为 None
        - exception: 失败时的异常对象，成功时为 None

    特性：
        - 单张失败不阻塞整个批次，异常通过元组返回
        - 输出顺序与输入顺序一致
        - 支持 generator/迭代器等惰性输入

    示例：
        >>> images = [cv2.imread(f) for f in file_list]
        >>> for idx, out, err in to_lineart_batch(images):
        ...     if err is None:
        ...         save(out, f"out_{idx}.png")
    """
    p = params or LineArtParams()
    n_workers = n_workers or _default_workers()
    n_workers = max(1, n_workers)

    # 转换为带索引的迭代器（支持 generator/迭代器）
    def _indexed_images():
        for i, img in enumerate(images):
            yield i, img

    # 预取任务（避免迭代器在多线程下竞争）
    tasks = list(_indexed_images())

    if not tasks:
        return

    results = {}
    exc_map = {}

    def _worker(idx, img):
        try:
            return idx, to_lineart(img, params=p, semantic_map=semantic_map), None
        except Exception as e:
            return idx, None, e

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker, idx, img): idx for idx, img in tasks}
        bar = _tqdm(futures, desc="lineart", total=len(futures), disable=not progress) if _HAS_TQDM and progress else futures
        for future in concurrent.futures.as_completed(bar):
            idx = futures[future]
            try:
                i, out, err = future.result()
                results[i] = out
                exc_map[i] = err
            except Exception as e:
                results[idx] = None
                exc_map[idx] = e

    # 按输入顺序产出
    for idx in range(len(tasks)):
        yield idx, results[idx], exc_map[idx]


def to_lineart_batch_simple(
    images: Union[Iterable, Iterator],
    params: Optional[Any] = None,
    n_workers: Optional[int] = None,
    progress: bool = False,
) -> Generator[Any, None, None]:
    """简化版批量 API：只产出结果，忽略异常（失败项返回 None）。

    参数：
        images: 可迭代序列
        params: LineArtParams 实例
        n_workers: 并行线程数
        progress: 是否显示进度条

    产出：
        每个输入图像的线稿结果，失败时为 None。
        输出顺序与输入顺序一致。

    示例：
        >>> results = list(to_lineart_batch_simple(images))
    """
    for idx, out, err in to_lineart_batch(
        images, params=params, n_workers=n_workers, progress=progress
    ):
        yield out
