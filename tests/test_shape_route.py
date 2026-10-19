"""shape 外形保真渲染路线测试（v1.10.0）。

核心契约：
- 默认 render_mode="splat"，默认管线输出与 v1.9.x 逐位一致（零回归）
- shape 路线：外形 IoU / 宽度保真 / 中心位置精度全面达标
- 墨判定：输出 < 128 为墨（背景为白）
"""
import cv2
import numpy as np
import pytest

from lineart import LineArtParams, to_lineart


def _vline(cols, h=192, w=192, color=40):
    """竖线：精确指定列集合（避免 round 语义歧义）。"""
    img = np.full((h, w, 3), 245, np.uint8)
    gt = np.zeros((h, w), dtype=bool)
    gt[:, list(cols)] = True
    img[gt] = color
    return img, gt


def _taper(h=192, w=192, cx=96.0, w0=2.0, w1=8.0, color=40):
    img = np.full((h, w, 3), 245, np.uint8)
    gt = np.zeros((h, w), dtype=bool)
    for y in range(h):
        t = y / (h - 1)
        width = w0 + (w1 - w0) * t
        lo = int(np.ceil(cx - width / 2))
        hi = int(np.floor(cx + width / 2))
        if hi >= lo:
            gt[y, max(lo, 0):min(hi + 1, w)] = True
    img[gt] = color
    return img, gt


def _ink(out):
    return out < 128


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / u if u else 0.0


def _row_width(mask, row):
    return int(mask[row].sum())


SPLAT = dict(render_mode="splat")
SHAPE = dict(render_mode="shape")


def test_default_render_mode_is_splat():
    assert LineArtParams().render_mode == "splat"
    assert LineArtParams().shape_ink == "response"
    assert LineArtParams().shape_closing == 1


def test_default_pipeline_bitwise_unchanged():
    """默认参数输出与显式 splat 逐位一致（v1.9.x 零回归）。"""
    img, _ = _vline(range(94, 98))
    out_default = to_lineart(img, LineArtParams())
    out_splat = to_lineart(img, LineArtParams(render_mode="splat"))
    assert np.array_equal(out_default, out_splat)


def test_shape_output_contract():
    img, _ = _vline(range(94, 98))
    out = to_lineart(img, LineArtParams(render_mode="shape"))
    assert out.shape == (192, 192)
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255
    assert out[0, 0] == 255  # 背景白


def test_shape_narrow_line_fidelity():
    """4px 硬边线：IoU ≥ 0.9，宽度 4±1。"""
    img, gt = _vline(range(94, 98))
    out = to_lineart(img, LineArtParams(render_mode="shape"))
    m = _ink(out)
    assert _iou(m, gt) >= 0.9
    assert abs(_row_width(m, 96) - 4) <= 1


def test_shape_wide_line_beats_splat():
    """10px 宽线（超 merge_gap=6，splat 边缘配对失败退化为双线）：
    shape IoU ≥ 0.9 且显著优于 splat。"""
    img, gt = _vline(range(91, 101))
    out_shape = to_lineart(img, LineArtParams(render_mode="shape"))
    out_splat = to_lineart(img, LineArtParams(render_mode="splat"))
    iou_shape = _iou(_ink(out_shape), gt)
    iou_splat = _iou(_ink(out_splat), gt)
    assert iou_shape >= 0.9
    assert iou_shape > iou_splat
    assert abs(_row_width(_ink(out_shape), 96) - 10) <= 1


def test_shape_taper_width_follows_source():
    """变宽线 2→8px：逐行宽度跟随原图（corr ≥ 0.9，MAE ≤ 0.5px）。"""
    img, gt = _taper()
    out = to_lineart(img, LineArtParams(render_mode="shape"))
    prof = _ink(out).sum(axis=1).astype(np.float64)
    gt_prof = gt.sum(axis=1).astype(np.float64)
    corr = float(np.corrcoef(prof, gt_prof)[0, 1])
    mae = float(np.abs(prof - gt_prof).mean())
    assert corr >= 0.9
    assert mae <= 0.5


def test_shape_circle_position_accuracy():
    """模糊圆 r=40：径向误差 ≤ 0.15px（外形半深度切分的面积无偏性）。"""
    img = np.full((192, 192, 3), 245, np.uint8)
    cv2.circle(img, (96, 96), 40, (60,) * 3, 2, cv2.LINE_AA)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    out = to_lineart(img, LineArtParams(render_mode="shape"))
    ys, xs = np.nonzero(_ink(out))
    assert len(xs) > 100
    d = np.hypot(xs - 96.0, ys - 96.0)
    assert abs(d.mean() - 40.0) <= 0.15


def test_shape_ink_modes_all_valid():
    img, _ = _vline(range(94, 98))
    for mode in ("response", "source", "flat"):
        out = to_lineart(img, LineArtParams(render_mode="shape",
                                            shape_ink=mode))
        assert out.shape == (192, 192)
        assert out.dtype == np.uint8
    # flat：掩膜内均匀黑 → 线像素非白即背景白
    out = to_lineart(img, LineArtParams(render_mode="shape",
                                        shape_ink="flat"))
    vals = set(np.unique(out).tolist())
    assert vals <= {0, 255} or all(v in (0, 255) or v > 200 for v in vals)


def test_shape_binary_mode():
    img, _ = _vline(range(94, 98))
    out = to_lineart(img, LineArtParams(render_mode="shape",
                                        mode="binary"))
    assert set(np.unique(out).tolist()) <= {0, 255}
    assert (_ink(out)).any()


def test_shape_semantic_selectors():
    img, _ = _vline(range(94, 98))
    for sem in ("xdog", "adaptive"):
        out = to_lineart(img, LineArtParams(render_mode="shape",
                                            semantic=sem))
        assert out.shape == (192, 192)


def test_shape_invalid_ink_mode_rejected():
    img, _ = _vline(range(94, 98))
    with pytest.raises(ValueError):
        to_lineart(img, LineArtParams(render_mode="shape",
                                      shape_ink="unknown"))


def test_shape_blank_image_clean():
    """空白图：无墨输出（滞回无强种子 → 掩膜空）。"""
    img = np.full((128, 128, 3), 250, np.uint8)
    out = to_lineart(img, LineArtParams(render_mode="shape"))
    assert not (_ink(out)).any()
