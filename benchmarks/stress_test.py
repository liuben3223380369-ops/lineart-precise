"""压力测试：尺寸 / 内容 / dtype / 参数 / 组合 / 内存 六维度。

不追求"全部通过"，而是暴露崩溃、异常、超时、内存泄漏的真实边界，
供逐项修复。每个场景独立 try/except，失败不阻断后续。
"""
import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import time
import resource
import traceback

import cv2
import numpy as np

from lineart import LineArtParams, to_lineart

RESULTS = []


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def case(name, fn, timeout=60.0):
    t0 = time.perf_counter()
    tag = "PASS"
    detail = ""
    try:
        fn()
        dt = (time.perf_counter() - t0) * 1000
        detail = f"{dt:.0f}ms"
    except Exception as e:
        tag = "FAIL"
        detail = f"{type(e).__name__}: {str(e)[:160]}"
    RESULTS.append((tag, name, detail))
    return tag


def case_expect(name, fn, exc, timeout=60.0):
    """断言 fn 抛出指定异常类型（预期拒绝场景）。"""
    t0 = time.perf_counter()
    try:
        fn()
        tag, detail = "FAIL", "未抛出预期异常"
    except exc as e:
        tag, detail = "PASS", f"预期拒绝 {type(e).__name__}"
    except Exception as e:
        tag, detail = "FAIL", f"抛错类型不符 {type(e).__name__}: {str(e)[:100]}"
    RESULTS.append((tag, name, detail))
    return tag


def B(img=None, shape=(128, 128), color=255):
    """造 uint8 BGR 图。"""
    if img is not None:
        return img
    return np.full((*shape, 3), color, np.uint8)


# 内容生成器
def solid(v):      return np.full((128, 128, 3), v, np.uint8)
def noise():       return np.random.default_rng(0).integers(0, 256, (128, 128, 3), np.uint8)
def graded():      return np.tile(np.linspace(0, 255, 128, dtype=np.uint8)[None, :, None], (128, 1, 3))
def checker():
    img = np.zeros((128, 128, 3), np.uint8)
    img[::8, :] = 255; img[:, ::8] = 255
    return img


print("=" * 72)
print("A. 尺寸压力")
print("=" * 72)
case("1x1", lambda: to_lineart(B(shape=(1, 1))))
case("2x2", lambda: to_lineart(B(shape=(2, 2))))
case("3x7 极小", lambda: to_lineart(B(shape=(3, 7))))
case("1x512 线", lambda: to_lineart(B(shape=(1, 512))))
case("512x512", lambda: to_lineart(B(shape=(512, 512))))
case("4096x4096 满格", lambda: to_lineart(B(shape=(4096, 4096))))
case("8192x8192 超限", lambda: to_lineart(B(shape=(8192, 8192))))

print("=" * 72)
print("B. 内容压力")
print("=" * 72)
case("纯白", lambda: to_lineart(solid(255)))
case("纯黑", lambda: to_lineart(solid(0)))
case("中灰 128", lambda: to_lineart(solid(128)))
case("随机噪声", lambda: to_lineart(noise()))
case("水平渐变", lambda: to_lineart(graded()))
case("棋盘格", lambda: to_lineart(checker()))
case("纯白+auto_invert off", lambda: to_lineart(solid(255), LineArtParams(auto_invert=False)))

print("=" * 72)
print("C. dtype / 通道压力")
print("=" * 72)
def f32_img():
    return np.full((128, 128, 3), 255.0, np.float32)
def f64_img():
    return np.full((128, 128, 3), 255.0, np.float64)
def gray2d():
    return np.full((128, 128), 128, np.uint8)
def rgba():
    return np.full((128, 128, 4), 255, np.uint8)
case("float32 输入", lambda: to_lineart(f32_img()))
case("float64 输入", lambda: to_lineart(f64_img()))
case("二维灰度输入", lambda: to_lineart(gray2d()))
case("RGBA 输入", lambda: to_lineart(rgba()))
case("single_channel_3d", lambda: to_lineart(np.full((128, 128, 1), 255, np.uint8)))
case_expect("空图 0x0（预期拒绝）", lambda: to_lineart(np.zeros((0, 0, 3), np.uint8)), ValueError)
case_expect("None 输入（预期拒绝）", lambda: to_lineart(None), ValueError)
case_expect("1D 输入（预期拒绝）", lambda: to_lineart(np.zeros((128,), np.uint8)), ValueError)

print("=" * 72)
print("D. 参数压力（极端值）")
print("=" * 72)
base = checker()
case("thickness=0", lambda: to_lineart(base, LineArtParams(thickness=0)))
case("thickness=-1", lambda: to_lineart(base, LineArtParams(thickness=-1)))
case("thickness=50", lambda: to_lineart(base, LineArtParams(thickness=50)))
case("render_ss=0", lambda: to_lineart(base, LineArtParams(render_ss=0)))
case("render_ss=8", lambda: to_lineart(base, LineArtParams(render_ss=8)))
case("sharpen=2.0", lambda: to_lineart(base, LineArtParams(sharpen=2.0)))
case("sharpen=-1", lambda: to_lineart(base, LineArtParams(sharpen=-1)))
case("denoise=5", lambda: to_lineart(base, LineArtParams(denoise=5)))
case("detail=100", lambda: to_lineart(base, LineArtParams(detail=100)))
case("sensitivity=-5", lambda: to_lineart(base, LineArtParams(line_sensitivity=-5)))
case("smooth_radius=301", lambda: to_lineart(base, LineArtParams(smooth_radius=301)))
case("min_area=-10", lambda: to_lineart(base, LineArtParams(min_area=-10)))
case("max_side=1", lambda: to_lineart(base, LineArtParams(max_side=1)))
case("illumination=10", lambda: to_lineart(base, LineArtParams(illumination=10)))
case("gradient_method=未知", lambda: to_lineart(base, LineArtParams(gradient_method="nope")))
case_expect("subpixel_method=未知（预期拒绝）", lambda: to_lineart(base, LineArtParams(subpixel_method="nope")), ValueError)
case("semantic=未知", lambda: to_lineart(base, LineArtParams(semantic="nope")))
case("scene=未知", lambda: to_lineart(base, LineArtParams(scene="nope")))

print("=" * 72)
print("E. 语义×场景×输出 组合")
print("=" * 72)
for sem in ("fdog", "xdog", "adaptive"):
    for out in ("grayscale", "binary", "color", "layered"):
        case(f"{sem}/{out}", lambda s=sem, o=out: to_lineart(
            checker(), LineArtParams(semantic=s, output=o)))

print("=" * 72)
print("E2. shape 外形保真路线（render_mode=shape）")
print("=" * 72)
for ink_mode in ("response", "source", "flat"):
    case(f"shape/{ink_mode}", lambda k=ink_mode: to_lineart(
        checker(), LineArtParams(render_mode="shape", shape_ink=k)))
case("shape/binary", lambda: to_lineart(
    checker(), LineArtParams(render_mode="shape", mode="binary")))
case("shape+closing0", lambda: to_lineart(
    checker(), LineArtParams(render_mode="shape", shape_closing=0)))

print("=" * 72)
print("F. 内存稳定性（连续 30 次 512x512）")
print("=" * 72)
img = checker()
r0 = rss_mb()
for i in range(30):
    to_lineart(img)
r1 = rss_mb()
growth = r1 - r0
print(f"  RSS 变化: {r0:.1f}MB -> {r1:.1f}MB（增长 {growth:.1f}MB）", end="")
print("  [泄漏风险!]" if growth > 100 else "  [正常]")
RESULTS.append(("PASS" if growth <= 100 else "FAIL", "F.内存 30次循环", f"增长{growth:.1f}MB"))

print("=" * 72)
print("G. 超大图性能上限（4096x4096）")
print("=" * 72)
case("4096x4096 fdog", lambda: to_lineart(B(shape=(4096, 4096))), timeout=300)
case("4096x4096 shape", lambda: to_lineart(B(shape=(4096, 4096)), LineArtParams(render_mode="shape")), timeout=300)

# ── 汇总 ──
print()
print("=" * 72)
print("汇总")
print("=" * 72)
fails = [r for r in RESULTS if r[0] == "FAIL"]
for tag, name, detail in RESULTS:
    print(f"  [{tag}] {name:<28} {detail}")
print(f"\n总计 {len(RESULTS)} 项，失败 {len(fails)} 项")
if fails:
    print("\n失败清单：")
    for _, name, detail in fails:
        print(f"  - {name}: {detail}")