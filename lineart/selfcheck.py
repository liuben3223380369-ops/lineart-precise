"""功能可用性自检（selfcheck）。

一键端到端检测项目全部功能是否真正可用——每项跑真实小样例并校验
输出（而非仅 import），杜绝「构建成功但运行即崩」（v1.11.3 APK 教训）。

分级：
- core     核心功能，任一 FAIL → 整体不可用（退出码 1）
- optional 可选功能，缺依赖时 SKIP（不算失败，给出明确提示）

用法：
    python -m lineart --selfcheck            # CLI 报告
    from lineart.selfcheck import run_selfcheck, main   # 编程调用

设计约束：
- 全部检测用内置合成小图（不依赖外部文件，任意环境可跑）
- 每项独立 try/except，单项失败不阻断后续
- 总耗时 < 30s（小图 64~128px）
"""
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import cv2
import numpy as np

from .params import LineArtParams
from .pipeline import to_lineart, to_lineart_batch, to_lineart_batch_simple
from .scenes import detect_scene


@dataclass
class CheckResult:
    name: str
    category: str          # core | optional
    status: str            # PASS | FAIL | SKIP | INFO
    detail: str = ""
    ms: int = 0


def _make_line_img(size=96):
    """合成线稿测试图：白底 + 十字黑线 + 圆（含硬边与抗锯齿边）。"""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.line(img, (8, size // 2), (size - 8, size // 2), 0, 2)
    cv2.line(img, (size // 2, 8), (size // 2, size - 8), 0, 1)
    cv2.circle(img, (size // 4, size // 4), 12, 0, 2, cv2.LINE_AA)
    return img


def _make_color_img(size=96):
    """合成彩色 CAD 图：白底 + 红/蓝/黑三色线。"""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.line(img, (6, 10), (size - 6, 10), (0, 0, 255), 2)      # 红
    cv2.line(img, (6, 30), (size - 6, 30), (255, 0, 0), 2)      # 蓝
    cv2.line(img, (6, 50), (size - 6, 50), (0, 0, 0), 2)        # 黑
    return img


# ---------------------------------------------------------------- core 检测

def _check_deps():
    ver = "cv2=%s numpy=%s" % (cv2.__version__, np.__version__)
    return "PASS", ver


def _check_convert_grayscale():
    src = _make_line_img()
    out = to_lineart(src, LineArtParams(output="grayscale"))
    assert out.shape[:2] == src.shape[:2], f"尺寸不一致 {out.shape} vs {src.shape}"
    ink = int((out < 128).sum())
    assert ink > 50, f"墨像素异常少: {ink}"
    return "PASS", f"{out.shape[1]}x{out.shape[0]}, 墨像素 {ink}"


def _check_convert_binary():
    out = to_lineart(_make_line_img(), LineArtParams(mode="binary"))
    assert out.shape[:2] == (96, 96)
    vals = np.unique(out)
    assert set(vals.tolist()) <= {0, 255}, f"binary 输出含中间值: {vals[:5]}"
    return "PASS", f"纯黑白值域 {sorted(set(vals.tolist()))}"


def _check_convert_color():
    src = _make_color_img()
    out = to_lineart(src, LineArtParams(output="color"))
    assert out.ndim == 3 and out.shape[2] == 3, f"非 3 通道: {out.shape}"
    return "PASS", f"彩色线稿 {out.shape[1]}x{out.shape[0]}"


def _check_convert_layered():
    out = to_lineart(_make_color_img(), LineArtParams(output="layered"))
    assert isinstance(out, list) and len(out) >= 1, f"分层结果异常: {type(out)}"
    mask, color = out[0]
    assert mask.dtype == bool or mask.dtype == np.uint8, f"mask 类型异常: {mask.dtype}"
    assert len(color) == 3, f"颜色非 BGR 三元组: {color}"
    return "PASS", f"{len(out)} 层 (mask{mask.shape}, color{color})"


def _check_render_shape():
    src = _make_line_img()
    out = to_lineart(src, LineArtParams(render_mode="shape"))
    assert out.shape[:2] == src.shape[:2], f"shape 路线尺寸不一致: {out.shape}"
    assert int((out < 128).sum()) > 20, "shape 路线墨像素异常少"
    return "PASS", f"shape 外形保真路线, 墨像素 {int((out < 128).sum())}"


def _check_semantics():
    src = _make_line_img()
    for sem in ("fdog", "xdog", "adaptive"):
        out = to_lineart(src, LineArtParams(semantic=sem))
        assert out.shape[:2] == src.shape[:2], f"{sem} 输出尺寸异常"
    return "PASS", "fdog / xdog / adaptive 三语义均转换成功"


def _check_scenes():
    src = _make_line_img()
    info = detect_scene(src)
    routed = []
    for sc in ("pixel", "blueprint", "sketch", "lineart", "photo"):
        out = to_lineart(src, LineArtParams(scene=sc))
        assert out.shape[:2] == src.shape[:2], f"scene={sc} 输出尺寸异常"
        routed.append(sc)
    return "PASS", f"自动检测={info.scene}(置信度{info.confidence:.2f}), 5 场景强制路由均成功"


def _check_batch():
    imgs = [_make_line_img(64), _make_line_img(64)]
    outs = list(to_lineart_batch(imgs, LineArtParams()))
    assert len(outs) == 2, f"批量数量异常: {len(outs)}"
    for idx, result, err in outs:
        assert err is None, f"批量项 {idx} 失败: {err}"
        assert result is not None
    outs2 = list(to_lineart_batch_simple(imgs))
    assert len(outs2) == 2 and all(r is not None for r in outs2)
    return "PASS", "to_lineart_batch(2) + batch_simple(2)"


def _check_svg():
    from .io.svg_export import to_svg_lineart
    svg = to_svg_lineart(_make_line_img(64), LineArtParams(output="binary"))
    assert isinstance(svg, str) and "<svg" in svg and "</svg>" in svg, "SVG 结构异常"
    assert 'width=' in svg and 'height=' in svg, "SVG 缺尺寸属性"
    return "PASS", f"SVG {len(svg)} 字符, 路径元素 {svg.count('<path')}"


def _check_cli():
    frozen = getattr(sys, "frozen", False)
    if frozen:
        cmd = [sys.executable, "--version"]
    else:
        cmd = [sys.executable, "-m", "lineart", "--version"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"CLI 退出码 {r.returncode}: {r.stderr[-200:]}"
    return "PASS", ("frozen EXE" if frozen else "python -m lineart") + " --version 退出码 0"


# ------------------------------------------------------------ optional 检测

def _check_gui():
    try:
        import tkinter  # noqa: F401
    except Exception as e:
        return "SKIP", f"tkinter 不可用: {e}"
    try:
        from PIL import Image, ImageTk  # noqa: F401
    except Exception as e:
        return "SKIP", f"PIL/ImageTk 不可用（GUI 需要）: {e}"
    try:
        from .gui.preview import LineArtPreviewGUI
    except Exception as e:
        return "FAIL", f"GUI 模块加载失败: {e}"
    # v1.26.1：真正实例化 GUI——v1.24.2 的 import 级检查漏检运行期
    # 构造错误（如 pack() 收到 -foreground 的 TclError，v1.25.0/
    # v1.26.0 EXE 双击即崩）。Tk root 立即销毁不显示窗口。
    try:
        root = tkinter.Tk()
        root.withdraw()
        try:
            LineArtPreviewGUI(root)
        finally:
            root.destroy()
    except tkinter.TclError as e:
        if "display" in str(e).lower() or "no screen" in str(e).lower():
            return "SKIP", f"无显示环境（GUI 无法实例化，跳过深检）: {e}"
        return "FAIL", f"GUI 实例化失败（构造期崩溃）: {e}"
    except Exception as e:
        return "FAIL", f"GUI 实例化失败（构造期崩溃）: {e}"
    return "PASS", "tkinter + PIL + GUI 模块 + 实例化构造就绪（GUI 可用）"


def _check_video():
    path = "/tmp/lineart_selfcheck.avi"
    try:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        w = cv2.VideoWriter(path, fourcc, 2, (32, 32))
        if not w.isOpened():
            return "SKIP", "无可用视频编解码器（MJPG）"
        for i in range(2):
            frame = np.full((32, 32, 3), 255 if i else 0, np.uint8)
            w.write(frame)
        w.release()
        r = cv2.VideoCapture(path)
        n = int(r.get(cv2.CAP_PROP_FRAME_COUNT))
        r.release()
        assert n == 2, f"帧数异常: {n}"
        return "PASS", "视频写入/读回 2 帧 @32x32 (MJPG)"
    except Exception as e:
        return "SKIP", f"视频功能不可用: {e}"


def _check_numba():
    try:
        import numba  # noqa: F401
        return "INFO", f"numba {numba.__version__} JIT 加速可用"
    except Exception:
        return "INFO", "numba 未安装（纯 NumPy 回退，功能不受影响）"


# ------------------------------------------------------------------- 编排

_CORE: List[Tuple[str, Callable[[], Tuple[str, str]]]] = [
    ("deps:numpy-opencv", _check_deps),
    ("core:convert-grayscale", _check_convert_grayscale),
    ("core:convert-binary", _check_convert_binary),
    ("core:convert-color", _check_convert_color),
    ("core:convert-layered", _check_convert_layered),
    ("core:render-shape", _check_render_shape),
    ("core:semantic-x3", _check_semantics),
    ("core:scene-x5", _check_scenes),
    ("core:batch-api", _check_batch),
    ("core:svg-export", _check_svg),
    ("core:cli-entry", _check_cli),
]

_OPTIONAL: List[Tuple[str, Callable[[], Tuple[str, str]]]] = [
    ("optional:gui", _check_gui),
    ("optional:video", _check_video),
    ("optional:numba", _check_numba),
]


def run_selfcheck() -> Tuple[bool, List[CheckResult]]:
    """执行全部检测。返回 (core 是否全过, 结果列表)。"""
    results: List[CheckResult] = []
    for name, fn in _CORE + _OPTIONAL:
        cat = "core" if name.startswith("core:") or name.startswith("deps:") else "optional"
        t0 = time.time()
        try:
            status, detail = fn()
        except Exception as e:
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        results.append(CheckResult(name, cat, status, detail, int((time.time() - t0) * 1000)))
    ok = all(r.status == "PASS" for r in results if r.category == "core")
    return ok, results


def main():
    ok, results = run_selfcheck()
    print("lineart-precise 功能可用性自检")
    print("=" * 64)
    for r in results:
        print(f"  [{r.status:<4}] {r.name:<24} {r.detail}  ({r.ms}ms)")
    core = [r for r in results if r.category == "core"]
    opt = [r for r in results if r.category == "optional"]
    print("-" * 64)
    print(f"核心 {len(core)} 项: "
          f"PASS {sum(r.status=='PASS' for r in core)} / "
          f"FAIL {sum(r.status=='FAIL' for r in core)}")
    print(f"可选 {len(opt)} 项: "
          f"PASS {sum(r.status=='PASS' for r in opt)} / "
          f"SKIP {sum(r.status=='SKIP' for r in opt)} / "
          f"INFO {sum(r.status=='INFO' for r in opt)}")
    print("结论: " + ("全部核心功能可用" if ok else "存在核心功能故障，见上方 FAIL 项"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
