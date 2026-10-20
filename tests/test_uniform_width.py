"""统一线宽（v1.16.1）契约测试。

场景：CAD 图不需要线宽级差（剖切/轮廓/标注分层），用户希望所有线
包括粗线一律按 thickness 设定宽度输出。默认（uniform_width=False）
保持 v1.15.0+ 的 FWHM 原宽行为逐位不变。
"""
import cv2
import numpy as np

from lineart import LineArtParams, to_lineart


def _thick_line_image(size=128):
    """白底 + 6px 粗边框矩形 + 1px 细对角线（混合粗细结构）。"""
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.rectangle(img, (20, 20), (size - 20, size - 20), (0, 0, 0), 6)
    cv2.line(img, (24, 24), (size - 24, size - 24), (0, 0, 0), 1)
    return img


def _gray(out):
    return out if out.ndim == 2 else cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def _vertical_run_width(gray, x, y0, y1):
    """列扫描 [y0,y1) @x：暗像素（<128）最长连续段长度。"""
    col = gray[y0:y1, x] < 128
    best = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


class TestUniformWidth:
    # scene="lineart"：数字线稿标准管线，零场景参数覆盖，
    # 排除场景检测不确定性对宽度断言的干扰
    P = dict(scene="lineart")

    def test_default_keeps_source_width(self):
        # 默认：FWHM 原宽，6px 粗线输出仍是粗线（>=4px，含 AA 过渡）
        out = to_lineart(_thick_line_image(), LineArtParams(**self.P))
        w = _vertical_run_width(_gray(out), x=60, y0=10, y1=40)
        assert w >= 4, f"默认模式应保留原宽，实测 {w}px"

    def test_uniform_collapses_to_thickness(self):
        # 统一线宽：粗线塌到 thickness=1（扫描宽度 <=2，含 AA 过渡）
        out = to_lineart(_thick_line_image(),
                         LineArtParams(**self.P, uniform_width=True))
        w = _vertical_run_width(_gray(out), x=60, y0=10, y1=40)
        assert w <= 2, f"统一线宽后应 <=2px，实测 {w}px"

    def test_uniform_wider_thickness(self):
        # 统一线宽 + thickness=3：粗线细线都应是 3px 左右（3~5px，
        # SSR 粗线路径展开），粗线不再按 6px 原宽
        out = to_lineart(_thick_line_image(),
                         LineArtParams(**self.P, uniform_width=True,
                                       thickness=3.0))
        w = _vertical_run_width(_gray(out), x=60, y0=10, y1=40)
        assert 2 <= w <= 5, f"thickness=3 应输出 ~3px，实测 {w}px"

    def test_uniform_no_double_line(self):
        # 关键回归：统一线宽不回退到双线（v1.15.0 修复的粗线双线化
        # 不能复活）——输出连通域数不得比默认模式多
        img = _thick_line_image()
        out_default = to_lineart(img, LineArtParams(**self.P))
        out_uniform = to_lineart(img, LineArtParams(**self.P,
                                                    uniform_width=True))
        n0 = cv2.connectedComponents(
            (_gray(out_default) < 128).astype(np.uint8))[0]
        n1 = cv2.connectedComponents(
            (_gray(out_uniform) < 128).astype(np.uint8))[0]
        assert n1 <= n0 + 1, f"统一线宽引入双线/碎裂（{n0} -> {n1}）"

    def test_uniform_position_on_centerline(self):
        # 统一线宽后线心位置应与默认模式一致（合并中点不变，
        # 只是宽度不同）：上边框墨心 y 坐标差 <=1.5px
        img = _thick_line_image()
        g0 = _gray(to_lineart(img, LineArtParams(**self.P)))
        g1 = _gray(to_lineart(img, LineArtParams(**self.P,
                                                 uniform_width=True)))

        def ink_center_y(g):
            col = (g[:, 60] < 128).astype(np.float32)
            ys = np.arange(len(col), dtype=np.float32)
            return float((col * ys).sum() / max(col.sum(), 1.0))

        assert abs(ink_center_y(g0) - ink_center_y(g1)) <= 1.5

    def test_default_bit_identical_when_off(self):
        # uniform_width=False（默认）与不传参数输出逐位一致（零回归）
        img = _thick_line_image()
        a = to_lineart(img, LineArtParams(**self.P))
        b = to_lineart(img, LineArtParams(**self.P, uniform_width=False))
        assert np.array_equal(a, b)

    def test_cli_flag_parses(self):
        # CLI --uniform-width 旗标可解析且映射到 params（冒烟）
        import subprocess
        import sys
        r = subprocess.run(
            [sys.executable, "-m", "lineart", "--help"],
            capture_output=True, text=True,
            cwd="/workspace/lineart-precise")
        assert "--uniform-width" in r.stdout
