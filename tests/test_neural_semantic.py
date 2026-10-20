"""v1.17.0 神经语义（semantic="neural"）契约测试。

后处理链测试不依赖权重（CI 可跑）；推理与集成测试在权重存在时
才执行（本地/自备权重环境）。
"""
import cv2
import numpy as np
import pytest

from lineart import LineArtParams, to_lineart
from lineart.semantics import neural as N


def _line_ink(h=96, w=160, y0=48, sigma=1.5):
    """单条水平线的高斯剖面 ink（模拟神经输出：位置准、有拖尾）。"""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.exp(-((yy - y0) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


class TestNeuralPostprocess:
    """神经后处理链（无权重依赖）。"""

    def test_line_kept(self):
        ink = _line_ink()
        mask = N.neural_postprocess(ink, LineArtParams())
        assert mask.dtype == bool
        assert mask.any(), "明确线条响应被整体丢弃"
        rows = np.where(mask.any(axis=1))[0]
        assert abs(rows.mean() - 48) <= 1.5, f"线位置漂移 {rows.mean():.2f}"

    def test_strong_line_continuity(self):
        """贯穿全图的强线：桥接后逐列连续（断线修复契约）。"""
        ink = _line_ink()
        ink[:, 70:76] *= 0.55  # 制造弱连接断点
        mask = N.neural_postprocess(ink, LineArtParams())
        cols = mask.sum(axis=0)
        assert (cols > 0).mean() >= 0.95, f"强线列连续率 {(cols>0).mean():.2%}"

    def test_weak_background_removed(self):
        """低幅值响应（<=0.28，拉伸后低于滞回低阈）不自成线：
        仅左上强线段区域有输出，大片弱响应区无泄漏。"""
        rng = np.random.default_rng(7)
        ink = np.clip(rng.normal(0.06, 0.03, (96, 160)), 0, 1).astype(np.float32)
        ink[50:, :] = np.clip(rng.normal(0.24, 0.05, (46, 160)), 0, 1)
        ink[10:14, 10:80] = 1.0
        mask = N.neural_postprocess(ink, LineArtParams())
        assert mask[50:, :].sum() <= 2, "低幅值弱响应区泄漏成线"

    def test_output_is_thin_skeleton(self):
        """宽响应带（模拟神经拖尾）→ 骨架细线（<=2px）。"""
        ink = _line_ink(sigma=3.0)
        mask = N.neural_postprocess(ink, LineArtParams())
        widths = mask.sum(axis=0)[mask.sum(axis=0) > 0]
        assert widths.max() <= 2, f"输出线宽 {widths.max()}px（非骨架）"

    def test_small_blob_removed(self):
        """孤立小斑块（< min_area）：清理。"""
        ink = np.zeros((96, 160), np.float32)
        ink[10:14, 10:14] = 1.0
        mask = N.neural_postprocess(ink, LineArtParams())
        assert not mask[:30, :30].any(), "孤立小斑块未被清理"

    def test_min_area_passthrough(self):
        """min_area 参数透传：中等连通域（30px）在 min_area=50 下被删。"""
        ink = np.zeros((96, 160), np.float32)
        ink[10, 20:50] = 1.0
        mask10 = N.neural_postprocess(ink, LineArtParams(min_area=10))
        mask50 = N.neural_postprocess(ink, LineArtParams(min_area=50))
        assert mask10.any() and not mask50.any()


class TestNeuralSemanticErrors:
    """权重缺失时的报错契约（CI 无权重环境可跑）。"""

    def test_missing_model_file_raises(self):
        with pytest.raises(FileNotFoundError, match="lineart.onnx"):
            N.neural_semantic(np.zeros((16, 16, 3), np.uint8),
                              model_path="/nonexistent/path.onnx")

    def test_pipeline_error_mentions_setup(self, monkeypatch):
        """semantic=neural 且权重不可寻：报错含放置指引。"""
        monkeypatch.setattr(N, "_find_model", lambda mp=None: None)
        with pytest.raises(FileNotFoundError, match="LINEART_NEURAL_MODEL"):
            to_lineart(np.full((48, 64, 3), 255, np.uint8),
                       LineArtParams(semantic="neural"))


@pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
class TestNeuralSemanticInference:
    """ONNX 推理与管线集成（权重存在时执行）。"""

    def test_inference_contract(self):
        bgr = np.full((64, 80, 3), 200, np.uint8)
        cv2.circle(bgr, (40, 32), 18, (30, 30, 30), 2, cv2.LINE_AA)
        ink = N.neural_semantic(bgr)
        assert ink.shape == (64, 80)
        assert ink.dtype == np.float32
        assert 0.0 <= ink.min() and ink.max() <= 1.0
        assert ink.max() > 0.5, "圆形结构边缘无强响应"

    def test_pipeline_output_contract(self):
        bgr = np.full((96, 128, 3), 200, np.uint8)
        cv2.circle(bgr, (64, 48), 26, (30, 30, 30), 2, cv2.LINE_AA)
        for mode in ("antialiased", "binary"):
            out = to_lineart(bgr, LineArtParams(semantic="neural",
                                                mode=mode))
            assert out.shape[:2] == (96, 128), "输出尺寸=输入尺寸契约"
            assert out.dtype == np.uint8
            assert (out < 128).any(), "神经语义无线条输出"
            if mode == "binary":
                assert set(np.unique(out).tolist()) <= {0, 255}

    def test_neural_beats_builtin_on_photo_scene(self):
        """照片感场景（渐变+噪声背景中的弱对比山脊）：神经召回显著
        高于内建 FDoG（核心价值契约；干净背景下二者相当）。"""
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "demo",
                            "samples", "photo_like.png")
        if not os.path.exists(path):
            pytest.skip("demo 样本不存在")
        img = cv2.imread(path)
        h, w = img.shape[:2]
        x = np.arange(w)
        ridge = h * 0.62 + 36 * np.sin(x / 61.0) + 14 * np.sin(x / 23.0 + 2)

        def cover(out):
            m = out < 128
            hits = 0
            for xi in range(w):
                ys = np.where(m[:, xi])[0]
                if len(ys) and np.min(np.abs(ys - ridge[xi])) <= 2.5:
                    hits += 1
            return hits / w

        c_fdog = cover(to_lineart(img, LineArtParams(semantic="fdog",
                                                     scene="photo")))
        c_neural = cover(to_lineart(img, LineArtParams(semantic="neural",
                                                       scene="photo")))
        assert c_neural > 0.6, f"神经召回不足 {c_neural:.1%}"
        assert c_neural > c_fdog + 0.4, (
            f"神经相对内建无优势：neural={c_neural:.1%} fdog={c_fdog:.1%}")
