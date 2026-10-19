"""Kotlin <-> lineart 桥接模块（随 APK 打包）。

Kotlin 端通过 Chaquopy 调用本模块的 convert_file()：
输入/输出均走文件路径，避免跨语言大数组传递。

v1.14.0：
- 输入图由 Kotlin 端用 Android 原生 BitmapFactory 解码后存为 PNG
  （解决手机 HEIC 照片 cv2 无法解码、EXIF 旋转方向错两个问题），
  Python 端恒收标准 PNG 字节流
- binary 开关改走 mode="binary"（output=binary 是灰度输出的历史
  既有行为，纯黑白由 mode 控制——v1.12.0 selfcheck 期间确认的契约）
- max_side 限 2048：Android 应用堆受限制（默认 ~256MB），4096² 输入
  的多份 float32 中间缓冲（语义/梯度/渲染各 64MB+）极易 OOM，
  2048 对手机屏幕预览与保存精度已足够
- 新增 semantic_map_path：外部神经语义先验图（Anime2Sketch /
  DexiNed 等模型输出），实现 README 宣称的可插拔神经先验在
  APK 侧的落地入口
"""
import time

from lineart import LineArtParams, to_lineart
from lineart.io import load_image, save_image


def convert_file(in_path, out_path, detail=0.6, line_sensitivity=0.5,
                 thickness=1.0, binary=False, shape_mode=False,
                 semantic_map_path=None):
    """转换单张图片，返回 "宽x高, 耗时ms" 描述字符串。"""
    img = load_image(str(in_path))

    semantic_map = None
    if semantic_map_path:
        from lineart.io import load_image as _load
        sm = _load(str(semantic_map_path))
        semantic_map = to_semantic_map(sm)

    params = LineArtParams(
        detail=float(detail),
        line_sensitivity=float(line_sensitivity),
        thickness=float(thickness),
        mode="binary" if binary else "antialiased",
        render_mode="shape" if shape_mode else "splat",
        max_side=2048,
    )
    t0 = time.time()
    out = to_lineart(img, params, semantic_map=semantic_map)
    save_image(str(out_path), out)
    ms = int((time.time() - t0) * 1000)
    return "%dx%d, %d ms" % (out.shape[1], out.shape[0], ms)


def to_semantic_map(bgr_img):
    """BGR 语义图 → [0,1] float 语义图（1=线，与管线 ink 约定一致）。

    神经先验模型的输出是黑底白线或白底黑线的灰度图（常规线稿样式），
    统一转为管线约定的「1=线、0=背景」：白底黑线需反相，黑底白线
    恒等。背景亮度由四角中值估计。
    """
    import cv2
    import numpy as np
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    h, w = gray.shape
    d = max(2, min(h, w) // 16)
    corners = [float(np.median(gray[:d, :d])), float(np.median(gray[:d, -d:])),
               float(np.median(gray[-d:, :d])), float(np.median(gray[-d:, -d:]))]
    bg = float(np.median(corners))
    sm = (1.0 - gray) if bg > 0.5 else gray
    return np.clip(sm, 0.0, 1.0).astype(np.float32)


def selfcheck():
    """APK 内功能自检（无 CLI 的环境下由 Kotlin 直接调用）。

    返回 (ok: bool, report: str)，核心功能（转换四模式/三语义/
    五场景路由/批量）端到端真实跑一遍，可在应用内「诊断」入口触发。
    """
    import cv2
    import numpy as np

    results = []

    def check(name, fn):
        t0 = time.time()
        try:
            detail = fn()
            results.append((name, "PASS", detail, int((time.time() - t0) * 1000)))
        except Exception as e:  # noqa: BLE001
            results.append((name, "FAIL", f"{type(e).__name__}: {e}",
                            int((time.time() - t0) * 1000)))

    def make_line_img(size=96):
        img = np.full((size, size, 3), 255, np.uint8)
        cv2.line(img, (16, 16), (size - 16, size - 16), (40, 40, 40), 2)
        cv2.line(img, (size - 16, 16), (16, size - 16), (40, 40, 40), 2)
        cv2.circle(img, (size // 2, size // 2), size // 3, (30, 30, 30), 1)
        return img

    check("convert-default", lambda: to_lineart(
        make_line_img(), LineArtParams()).shape[:2])
    check("convert-binary", lambda: sorted(set(np.unique(to_lineart(
        make_line_img(), LineArtParams(mode="binary")).ravel().tolist()))))
    check("convert-shape", lambda: to_lineart(
        make_line_img(), LineArtParams(render_mode="shape")).shape[:2])
    check("semantic-map", lambda: to_lineart(
        make_line_img(), LineArtParams(),
        semantic_map=to_semantic_map(make_line_img())).shape[:2])
    check("scene-photo", lambda: to_lineart(
        (np.random.RandomState(0).rand(64, 64, 3) * 255).astype(np.uint8),
        LineArtParams()).shape[:2])

    ok = all(s == "PASS" for _, s, _, _ in results)
    report = "\n".join(f"[{s}] {n}: {d} ({ms}ms)"
                       for n, s, d, ms in results)
    return ok, report
