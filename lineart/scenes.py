"""场景自动检测：分析统计特征判断输入属于哪类场景，并给出参数覆盖。

支持的场景：
- "pixel"     像素艺术：少量颜色 + 2x2 块状结构 → 最近邻预放大
- "blueprint" 蓝图：蓝底白线 → 亮度反相成标准白底黑线
- "sketch"    拍照手绘/扫描文档：浅底 + 低饱和 + 背景低频不均 → adaptive 语义
- "lineart"   数字线稿（CAD/矢量渲染）：浅/深底 + 低饱和 + 背景均匀 → 标准管线
- "photo"     照片/自然图像：色彩丰富或渐变平滑 → FDoG 语义 + 纹理抑制

判别核心（sketch vs lineart）：
数字生成的图背景完美均匀（背景低频标准差≈0），
拍照/扫描的图背景必有阴影/渐变（背景低频标准差显著）。
"""
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class SceneInfo:
    scene: str
    confidence: float
    params: dict = field(default_factory=dict)
    pre: dict = field(default_factory=dict)


SCENE_PARAMS = {
    "pixel": {
        "semantic": "fdog",
        "denoise": 0.2,
        "illumination": 0.2,
        "min_area": 12,
        # 4x 空间渲染 4px 线，缩回原尺寸后恰为 1px（避免 AREA 缩回稀释线强度）
        "thickness": 4.0,
        # 4x 预放大后块边缘对间距 4px 落在合并窗口内，但块状硬边
        # 语义依赖边缘对渲染，pixel 场景保持原行为不合并
        "merge_edges": False,
    },
    "blueprint": {
        "semantic": "fdog",
        "denoise": 0.3,
        "illumination": 0.6,
        "min_area": 12,
    },
    "sketch": {
        "semantic": "adaptive",
        "illumination": 0.3,
        "denoise": 0.15,
        "min_area": 8,
        "line_sensitivity": 0.7,
    },
    "lineart": {},
    "photo": {
        "semantic": "fdog",
        "coherence": 0.75,
        "denoise": 0.55,
        "gradient_gamma": 0.6,
    },
}

SCENE_PRE = {
    "pixel": {"upscale": 4.0},
}


def _deep_core_ratio(small):
    """块状深核占比：位于「块状深核域」内的前景像素占比（v1.15.0）。

    深核 = 距任何背景 >=12px 的前景区域，只有大实心结构才有：
    色块拼图深核占比 0.53+、实心矩形 0.74、独立 40px 色块 0.20；
    一切**线状结构**的深核只是细条带芯：24px 线 0.04、细线 0。

    超粗结构（40px+ 线/环）的深核占比可达 0.45，纯占比判据无法
    与色块区分，追加**深核域形状**过滤（对深核掩码做连通域分析）：
    - 超粗线的深核是细长芯（细长度 >=8）→ 排除
    - 超粗环/框的深核是环形芯带（填充率 <0.3）→ 排除
    - 色块/色块粘连群的深核是块状（细长度 <8 且填充率 >=0.3）→ 通过

    替代 v1.13.0 的「DT>1.5 全局占比」（对粗线稿误判：12px 线 0.67
    → 误判 pixel → merge_edges=False → 全线宽双线，v1.14.0 APK
    实测）。阈值 0.15：色块下限 0.20、pixel_art.png 0.64、线状
    结构形状排除后为 0。宁漏判（小色块图无 4x 预放大，输出质量
    不变）不可误判（pixel 场景关边缘合并 → 双线）。
    """
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    vals, counts = np.unique(gray, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    fg = (np.abs(gray.astype(np.int16) - bg) > 40).astype(np.uint8)
    n = int(fg.sum())
    if n < 16:
        return 0.0
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    core = dist >= 12
    if not core.any():
        return 0.0
    if float(core.sum()) / n < 0.05:
        return 0.0  # 快速路径：无线状深核结构（细线/常规粗线稿）
    num, labels, st, _ = cv2.connectedComponentsWithStats(
        core.astype(np.uint8), connectivity=8)
    blocky_core = 0
    for i in range(1, num):
        area = int(st[i, cv2.CC_STAT_AREA])
        if area < 9:
            continue
        bw = int(st[i, cv2.CC_STAT_WIDTH])
        bh = int(st[i, cv2.CC_STAT_HEIGHT])
        if max(bw, bh) >= 8.0 * max(1, min(bw, bh)):
            continue  # 细长芯线（超粗线笔画）
        if area < 0.3 * bw * bh:
            continue  # 环形芯带（超粗环/框）
        blocky_core += area
    return float(blocky_core) / n


def _is_pixel_art(img_bgr, max_side=384):
    """像素艺术检测：颜色少 + 无渐变中间调（硬边块状结构）。

    判据（实测标定，见 demo/samples 统计）：
    - 量化到 16 级/通道后唯一颜色 ≤ 32
    - 逐通道相邻差（取 max）中，渐变差(8~90) 占比 < 0.35（抗锯齿线稿 ≥0.44）
    - 硬跳变(>160) 占比 > 0.25（排除平滑渐变照片）

    缩放必须用 INTER_NEAREST：INTER_AREA 会把硬边模糊成渐变导致漏检。
    """
    small = _shrink_nn(img_bgr, max_side)
    h, w = small.shape[:2]
    if h < 8 or w < 8:
        return 0.0

    q = (small.astype(np.int32) >> 4)  # 量化到 16 级/通道
    # 唯一颜色计数：3×4bit 打包成 int32 后 bincount（O(N) 计数），
    # 替代 np.unique(axis=0) 的 void-view 排序（大图下 10x+ 提速）
    key = (q[:, :, 0] << 8) | (q[:, :, 1] << 4) | q[:, :, 2]
    n_colors = int(np.count_nonzero(np.bincount(key.ravel(), minlength=4096)))
    if n_colors > 32:
        return 0.0

    # 深核占比：排除一切「线状结构」前景（细线/粗线/环/框的深核
    # 占比近零），只有大实心色块拼图（>=24px 块）才通过。
    # v1.13.0 用 DT>1.5 全局占比（0.5 阈值）只能排除 1~3px 细线，
    # 粗线稿（12px 线 0.67）漏判 pixel 导致全线宽双线（v1.15.0 教训）。
    if _deep_core_ratio(small) < 0.15:
        return 0.0

    f = small.astype(np.int16)
    dh = np.abs(np.diff(f, axis=1)).max(axis=2)
    dv = np.abs(np.diff(f, axis=0)).max(axis=2)
    d = np.concatenate([dh.ravel(), dv.ravel()])
    nz = d[d > 4]
    if len(nz) < 16:
        return 0.0
    mid_ratio = float(((nz >= 8) & (nz <= 90)).mean())
    hard_ratio = float((nz > 160).mean())
    if mid_ratio >= 0.35 or hard_ratio < 0.25:
        return 0.0

    # 单一长直边排除：双色调半平面/大色块图（每条扫描线的跳变位置
    # 几乎全部相同）是 1D 边缘结构而非像素艺术的 2D 块状图案，
    # 误判会触发无意义的 4x 预放大并破坏亚像素精化。
    if _single_straight_edge(dh > 4) or _single_straight_edge(dv > 4):
        return 0.0

    conf = min(1.0, (0.35 - mid_ratio) * 2.0 + hard_ratio)
    return min(1.0, conf)


def _single_straight_edge(jump):
    """判断逐行跳变掩码是否呈"单一长直边"（扫描线跳变位置高度一致）。

    像素艺术的行/列跳变位置随图案 2D 变化；半平面/单一大边缘图
    每条扫描线只在几乎相同的位置跳变 1~2 次（1D 结构）。
    jump: bool 数组，axis=1 为扫描方向。
    """
    h = jump.shape[0]
    if h < 8:
        return False
    counts = jump.sum(axis=1)
    active = counts > 0
    # 长直边应贯穿大部分扫描线
    if active.mean() < 0.8:
        return False
    # 2D 图案每条扫描线有多次跳变
    if counts[active].mean() > 2.5:
        return False
    first = np.argmax(jump, axis=1).astype(np.float64)
    return bool(first[active].std() < 1.5)


def _is_blueprint(img_bgr):
    """蓝图检测：蓝色主导 + 存在亮线（白/浅色笔迹）。

    OpenCV H∈[0,180)：蓝 ≈ [95,135]（对应色相 190°~270°）。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    blue = (h_ch >= 95) & (h_ch <= 135) & (s_ch > 70) & (v_ch > 30)
    blue_ratio = blue.mean()

    bright_line = (v_ch > 170) & (s_ch < 90)
    line_ratio = bright_line.mean()

    if blue_ratio > 0.30 and line_ratio > 0.015:
        conf = min(1.0, blue_ratio / 0.5 * 0.7 + line_ratio / 0.10 * 0.3)
        return min(1.0, conf)
    return 0.0


def _shrink(img_bgr, max_side):
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img_bgr
    tw, th = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    # 大比率（>4x）分两步 AREA：区域平均是线性算子，两级级联与单级
    # 期望一致（仅边界处理不同），但 4096→320 直缩时每输出像素需累积
    # ~165 个输入像素，两级先聚合到 1/4 可省近一半读带宽
    cur = img_bgr
    while max(cur.shape[:2]) > max_side * 4 and min(cur.shape[:2]) > 4:
        cw = max(4, cur.shape[1] // 4)
        ch = max(4, cur.shape[0] // 4)
        cur = cv2.resize(cur, (cw, ch), interpolation=cv2.INTER_AREA)
    return cv2.resize(cur, (tw, th), interpolation=cv2.INTER_AREA)


def _shrink_nn(img_bgr, max_side):
    """最近邻缩放：保持硬边不引入渐变（像素艺术检测专用）。"""
    h, w = img_bgr.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img_bgr
    return cv2.resize(img_bgr, (max(1, int(round(w * scale))),
                                 max(1, int(round(h * scale)))),
                      interpolation=cv2.INTER_NEAREST)


def _bg_unevenness(gray_u8):
    """背景低频不均匀度：拍照/扫描图有阴影（大），数字生成图背景平坦（≈0）。

    两层防御：
    1. 大核形态学闭运算抹掉线条，避免线条污染背景估计
    2. 只统计亮于全图 55 分位的"背景候选"像素，排除大暗块（圆盘/矩形前景）
       —— 大暗块残留会让 close 后的场出现假 std（圆盘 r=60 在 38px 核下填不掉）
    """
    g = gray_u8.astype(np.float32) / 255.0
    h, w = g.shape
    k = int(max(7, round(min(h, w) * 0.12))) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), max(h, w) * 0.03)
    mask = g > np.percentile(g, 55.0)
    if mask.sum() < 32:
        return float(bg.std())
    return float(bg[mask].std())


def detect_scene(img_bgr):
    """输入 BGR/灰度图像，返回 SceneInfo。"""
    if img_bgr.ndim == 2:
        img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)

    small = _shrink(img_bgr, 320)

    # 蓝图优先于像素艺术：蓝底白线彩色图颜色数少，容易被误判为像素图
    conf = _is_blueprint(small)
    if conf > 0.3:
        return SceneInfo("blueprint", conf, dict(SCENE_PARAMS["blueprint"]), {})

    conf = _is_pixel_art(img_bgr)
    if conf > 0.3:
        return SceneInfo("pixel", conf, dict(SCENE_PARAMS["pixel"]),
                         dict(SCENE_PRE["pixel"]))

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1].astype(np.float32) / 255.0
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # 四角中值（抗角落线条污染）：拍照阴影下四角亮度显著不同
    h, w = gray.shape
    dy, dx = max(2, h // 12), max(2, w // 12)
    cvals = [float(np.median(gray[:dy, :dx])),
             float(np.median(gray[:dy, -dx:])),
             float(np.median(gray[-dy:, :dx])),
             float(np.median(gray[-dy:, -dx:]))]
    corner_level = float(np.median(cvals))
    corner_spread = float(np.std(cvals))

    sat_mean = float(s_ch.mean())
    # 线条像素比例：相对背景暗 35 级（铅笔弱线灰度 120~160，绝对阈值会漏检）
    # 手绘/文档的线细 → <0.12；大暗块图形（形状图）→ 0.2+，不算"线稿照片"
    dark_ratio = float((gray < corner_level - 35.0).mean())

    # 彩色像素比例（照片特征）：通道间差异大
    f = small.astype(np.int16)
    chroma = f.max(axis=2) - f.min(axis=2)
    colorful = float((chroma > 24).mean())

    # 渐变平滑度：直方图相邻 bin 差异小 → 连续色调 → 照片；线稿直方图有尖峰
    hist = cv2.calcHist([gray], [0], None, [32], [0, 256]).ravel()
    hist = hist / max(hist.sum(), 1)
    smooth = 1.0 - float(np.abs(np.diff(hist)).mean() * 32.0)

    bg_std = _bg_unevenness(gray)

    # 黑底数字线稿（CAD 黑底截图）：四角暗 + 整体暗 → 标准管线(auto_invert 处理)
    if corner_level < 60.0 and float(gray.mean()) < 90.0:
        return SceneInfo("lineart", 0.8, dict(SCENE_PARAMS["lineart"]), {})

    light_bg = corner_level > 150.0
    low_sat = sat_mean < 0.16
    thin_lines = 0.003 <= dark_ratio <= 0.12
    not_photo_tone = smooth < 0.90

    if light_bg and low_sat and not_photo_tone:
        # 细线 + 背景不均匀（四角阴影 或 整体低频起伏）→ 拍照手绘/扫描文档
        if thin_lines and (corner_spread > 15.0 or bg_std > 0.045):
            conf = min(0.95, 0.45 + corner_spread / 40.0 + bg_std / 0.12)
            return SceneInfo("sketch", conf, dict(SCENE_PARAMS["sketch"]), {})
        conf = 0.6 if corner_spread < 10.0 else 0.5
        return SceneInfo("lineart", conf, dict(SCENE_PARAMS["lineart"]), {})

    if colorful > 0.30 or sat_mean > 0.22 or smooth > 0.86:
        conf = 0.5 + 0.4 * min(1.0, colorful / 0.6 + sat_mean)
        return SceneInfo("photo", min(0.95, conf),
                         dict(SCENE_PARAMS["photo"]), {})

    if dark_ratio > 0.5 and not light_bg:
        return SceneInfo("lineart", 0.6, dict(SCENE_PARAMS["lineart"]), {})

    return SceneInfo("photo", 0.4, dict(SCENE_PARAMS["photo"]), {})


def scene_info_for(scene):
    """按显式场景名构造 SceneInfo（用户手动指定时）。"""
    scene = scene if scene in SCENE_PARAMS else "lineart"
    return SceneInfo(scene, 1.0, dict(SCENE_PARAMS[scene]),
                     dict(SCENE_PRE.get(scene, {})))
