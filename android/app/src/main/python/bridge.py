"""Kotlin <-> lineart 桥接模块（随 APK 打包）。

Kotlin 端通过 Chaquopy 调用本模块的 convert_file()：
输入/输出均走文件路径，避免跨语言大数组传递。
"""
import time

from lineart import LineArtParams, to_lineart
from lineart.io import load_image, save_image


def convert_file(in_path, out_path, detail=0.6, line_sensitivity=0.5,
                 thickness=1.0, binary=False, shape_mode=False):
    """转换单张图片，返回 "宽x高, 耗时ms" 描述字符串。"""
    img = load_image(str(in_path))
    params = LineArtParams(
        detail=float(detail),
        line_sensitivity=float(line_sensitivity),
        thickness=float(thickness),
        output="binary" if binary else "grayscale",
        render_mode="shape" if shape_mode else "splat",
    )
    t0 = time.time()
    out = to_lineart(img, params)
    save_image(str(out_path), out)
    ms = int((time.time() - t0) * 1000)
    return "%dx%d, %d ms" % (out.shape[1], out.shape[0], ms)
