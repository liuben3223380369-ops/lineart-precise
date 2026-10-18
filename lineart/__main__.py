"""命令行入口：python -m lineart <input...> -o <output>"""
import argparse
import os
import sys
import time

from .io import load_image, save_image
from .localization.refine import gaussian_fit_subpixel
from .params import LineArtParams
from .pipeline import to_lineart


def _build_parser():
    ap = argparse.ArgumentParser(
        prog="lineart",
        description="高精度图片转线稿（CAD/线稿/照片通用：多尺度 XDoG/FDoG + 梯度验证 + 亚像素精化）")
    ap.add_argument("inputs", nargs="+", help="输入图像或目录")
    ap.add_argument("-o", "--output", required=True, help="输出目录或文件路径")
    ap.add_argument("--mode", choices=["antialiased", "binary"],
                    default="antialiased", help="抗锯齿灰度线 / 纯黑白线")
    ap.add_argument("--out", dest="output_mode",
                    choices=["grayscale", "color", "layered", "binary"],
                    default="grayscale",
                    help="输出模式：grayscale=白底黑线 / color=白底彩色线(CAD) / layered=分层输出 / binary=纯黑白")
    ap.add_argument("--semantic", choices=["fdog", "xdog", "adaptive"],
                    default="fdog",
                    help="语义选择器：fdog=流线连贯(Kang CLD) / xdog=经典多尺度 / "
                         "adaptive=Sauvola自适应阈值(拍照手绘/扫描文档)")
    ap.add_argument("--scene",
                    choices=["auto", "off", "pixel", "blueprint", "sketch",
                             "lineart", "photo"],
                    default="auto",
                    help="场景模式：auto=自动检测并适配(像素图预放大/蓝图反相/"
                         "手绘走adaptive语义/照片纹理抑制)；off=不启用")
    ap.add_argument("--semantic-map",
                    help="外部神经语义先验图（Anime2Sketch/DexiNed 输出），"
                         "提供后覆盖内建语义")
    ap.add_argument("--coherence", type=float, default=0.6,
                    help="线条连贯强度 0~1，越高线条越连贯、纹理越少")
    ap.add_argument("--smooth-radius", type=int, default=3,
                    help="导向滤波保边平滑半径（像素）")
    ap.add_argument("--detail", type=float, default=0.6,
                    help="细节强度 0~1，越高细节线条越多")
    ap.add_argument("--sensitivity", type=float, default=0.5,
                    help="线条灵敏度 0~1，越高越容易画线")
    ap.add_argument("--thickness", type=float, default=1.0,
                    help="线条粗细，1 为原尺寸")
    ap.add_argument("--denoise", type=float, default=0.4,
                    help="去噪强度 0~1")
    ap.add_argument("--illumination", type=float, default=0.8,
                    help="光照归一化强度 0~1")
    ap.add_argument("--min-area", type=int, default=10,
                    help="剔除小于该像素面积的噪点连通域")
    ap.add_argument("--max-side", type=int, default=4096,
                    help="输入长边上限，超出则等比缩小")
    ap.add_argument("--n-colors", type=int, default=8,
                    help="color/layered 模式的最大颜色数")
    ap.add_argument("--no-auto-invert", dest="auto_invert", action="store_false",
                    help="禁用自动底色检测与反转（默认对黑底图自动反转）")
    ap.add_argument("--remove-grid", action="store_true",
                    help="去除 CAD 网格/点阵背景（FFT 频域陷波）")
    ap.add_argument("--grid-period-min", type=int, default=2,
                    help="网格最小周期（像素），配合 --remove-grid")
    ap.add_argument("--remove-text", action="store_true",
                    help="检测并去除文字标注区域（MSER + inpainting）")
    ap.add_argument("--extract-centerline", action="store_true",
                    help="粗线条中心线提取（Zhang-Suen 细化）")
    ap.add_argument("--sharpen", type=float, default=0.0,
                    help="相干冲击锐化强度 0~1（模糊输入推荐 0.5~0.8，"
                         "只增强检测不影响定位精度）")
    ap.add_argument("--render-ss", dest="render_ss", type=int, default=2,
                    choices=[1, 2, 3, 4],
                    help="渲染超采样倍率（粗线宽度/边缘抗锯齿更准）")
    ap.add_argument("--no-consensus", dest="consensus", action="store_false",
                    help="关闭亚像素偏移邻域中值共识滤波")
    return ap


def _iter_files(paths):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    for path in paths:
        if os.path.isdir(path):
            for name in sorted(os.listdir(path)):
                if os.path.splitext(name)[1].lower() in exts:
                    yield os.path.join(path, name)
        else:
            yield path


def main(argv=None):
    args = _build_parser().parse_args(argv)
    semantic_map = None
    if args.semantic_map:
        from .semantics.neural import load_semantic_map
        semantic_map = load_semantic_map(args.semantic_map)
    params = LineArtParams(
        mode=args.mode,
        semantic=args.semantic,
        output=args.output_mode,
        detail=args.detail,
        line_sensitivity=args.sensitivity,
        thickness=args.thickness,
        denoise=args.denoise,
        illumination=args.illumination,
        coherence=args.coherence,
        smooth_radius=args.smooth_radius,
        min_area=args.min_area,
        max_side=args.max_side,
        n_colors=args.n_colors,
        remove_grid=args.remove_grid,
        grid_period_min=args.grid_period_min,
        remove_text=args.remove_text,
        extract_centerline=args.extract_centerline,
        auto_invert=args.auto_invert,
        scene=args.scene,
        sharpen=args.sharpen,
        render_ss=args.render_ss,
        consensus=args.consensus,
    )
    files = list(_iter_files(args.inputs))
    if not files:
        print("未找到输入图像", file=sys.stderr)
        return 1
    out_is_dir = os.path.isdir(args.output) or len(files) > 1
    if out_is_dir:
        os.makedirs(args.output, exist_ok=True)
    count = 0
    for path in files:
        t0 = time.time()
        img = load_image(path)
        result = to_lineart(img, params, semantic_map=semantic_map)
        if out_is_dir:
            stem = os.path.splitext(os.path.basename(path))[0]
            dst = os.path.join(args.output, stem + ".png")
        else:
            dst = args.output
        save_image(dst, result)
        dt = (time.time() - t0) * 1000
        print(f"[OK] {path} -> {dst}  ({img.shape[1]}x{img.shape[0]}, {dt:.0f} ms)")
        count += 1
    print(f"完成：{count} 张")
    return 0


if __name__ == "__main__":
    sys.exit(main())
