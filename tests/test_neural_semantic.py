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

    def test_output_band_contracted(self):
        """宽响应带（模拟神经拖尾）→ 滞回带不宽于响应带且逐列连续
        （v1.17.1 起不再骨架化：收缩由渲染端 pair_edge_mask 配对与
        FWHM 测宽完成，mask 域守住「不放大 + 不虚线」契约）。"""
        ink = _line_ink(sigma=3.0)
        mask = N.neural_postprocess(ink, LineArtParams())
        widths = mask.sum(axis=0)[mask.sum(axis=0) > 0]
        resp_width = int((ink > 0.2).sum(axis=1).max())
        assert widths.max() <= resp_width, \
            f"输出宽 {widths.max()}px 宽于响应宽 {resp_width}px（滞回放大）"
        cols = mask.sum(axis=0)
        assert (cols > 0).mean() >= 0.95, \
            f"输出列连续率 {(cols > 0).mean():.2%}（虚线化）"

    def test_double_ridge_pairing_contract(self):
        """粗线双边缘峰响应（中心谷底趋零）→ 滞回输出双脊线带且
        逐列连续（v1.17.1 设计契约：渲染端 pair_edge_mask 配对取中点，
        弧线连续性 58%→94% 的前提是脊线不断续）。"""
        h, w, y0 = 96, 160, 48
        yy = np.arange(h, dtype=np.float32)
        ridge_a = np.exp(-((yy - (y0 - 3)) ** 2) / 1.2)
        ridge_b = np.exp(-((yy - (y0 + 3)) ** 2) / 1.2)
        ink = np.maximum(ridge_a, ridge_b)[:, None].repeat(w, axis=1)
        ink[y0, :] = 0.01  # 中心谷底（模拟粗线神经响应）
        mask = N.neural_postprocess(ink, LineArtParams())
        assert mask.any(), "双脊线整体丢失"
        cols = mask.sum(axis=0)
        assert (cols > 0).mean() >= 0.95, \
            f"双脊线列连续率 {(cols > 0).mean():.2%}（虚线化）"
        widths = cols[cols > 0]
        assert 2 <= widths.max() <= 10, f"输出带宽异常 {widths.max()}px"

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

    def test_thick_arc_continuity_v118(self):
        """v1.18.0 弧线连续性契约：粗线（8px）弧在拍照级模糊下，
        双脊线交渲染端配对合并，弧带命中 ≥65%（v1.17.0 骨架化链
        实测 28%，重构后 79%）。"""
        rng = np.random.default_rng(5)
        img = np.ones((460, 520, 3), np.float32) * 0.95
        cv2.circle(img, (350, 210), 130, (0.35,) * 3, 8)
        cv2.line(img, (180, 330), (340, 330), (0.15,) * 3, 3)
        cv2.line(img, (260, 260), (260, 400), (0.15,) * 3, 3)
        img = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
        img = (img * 255).astype(np.uint8)
        out = to_lineart(img, LineArtParams(semantic="neural",
                                            mode="binary"))
        ink = out.max(axis=2) < 128 if out.ndim == 3 else out < 128
        hits = 0
        for a in range(0, 360, 2):
            rad = np.deg2rad(a)
            ok = any(ink[int(round(210 + (130 + dr) * np.sin(rad))),
                         int(round(350 + (130 + dr) * np.cos(rad)))]
                     for dr in range(-2, 3))
            hits += ok
        assert hits / 180 >= 0.65, f"弧带命中仅 {hits/180:.1%}（虚线化回归）"

    def test_cross_center_no_hole_v118(self):
        """v1.18.0 交叉白洞契约：拍照级模糊下十字交叉中心不出现
        空洞（v1.17.0 骨架化链实测有白洞）。"""
        rng = np.random.default_rng(5)
        img = np.ones((460, 520, 3), np.float32) * 0.95
        cv2.rectangle(img, (60, 60), (240, 180), (0.15,) * 3, 3)
        cv2.circle(img, (350, 210), 130, (0.35,) * 3, 8)
        cv2.line(img, (180, 330), (340, 330), (0.15,) * 3, 3)
        cv2.line(img, (260, 260), (260, 400), (0.15,) * 3, 3)
        img = cv2.GaussianBlur(img, (0, 0), 2.0)
        img = np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1)
        img = (img * 255).astype(np.uint8)
        out = to_lineart(img, LineArtParams(semantic="neural",
                                            mode="binary"))
        ink = out.max(axis=2) < 128 if out.ndim == 3 else out < 128
        assert ink[328:333, 258:263].any(), "交叉中心出现白洞（回归）"

    def test_semantic_map_path_shares_neural_chain(self):
        """外部语义图路径共享神经链：Kotlin 侧推理产物（黑线白底
        ink PNG）经 to_semantic_map 转换后与直连 neural 输出一致
        （v1.18.0 Android 神经先验桥接契约，IoU ≥0.8）。
        v1.18.0 起两路径整链共享：预处理链与背景归一化同样跳过
        ——保边去噪软化的过渡带会经 SSR 的 FWHM 宽度展开把线宽
        膨胀（实测 2px 线渲染成 6px，IoU 0.614）。"""
        import os
        bgr = np.full((200, 260, 3), 255, np.uint8)
        cv2.circle(bgr, (130, 100), 60, (60,) * 3, 8, cv2.LINE_AA)
        cv2.line(bgr, (40, 160), (220, 160), (50,) * 3, 3)
        ink = N.neural_semantic(bgr)
        ink_png = (np.clip(1.0 - ink, 0, 1) * 255).astype(np.uint8)
        tmp = os.path.join(os.path.dirname(__file__), "_neural_ink_tmp.png")
        cv2.imwrite(tmp, ink_png)
        try:
            sm = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            sm = (255.0 - sm) / 255.0
            out_direct = to_lineart(bgr, LineArtParams(semantic="neural",
                                                       mode="binary"))
            out_map = to_lineart(bgr, LineArtParams(mode="binary"),
                                 semantic_map=sm)
            d = out_direct.max(axis=2) < 128 if out_direct.ndim == 3 \
                else out_direct < 128
            m = out_map.max(axis=2) < 128 if out_map.ndim == 3 \
                else out_map < 128
            inter = (d & m).sum()
            union = (d | m).sum()
            assert union and inter / union >= 0.8, (
                f"语义图路径与直连路径 IoU 仅 {inter/union:.3f}")
        finally:
            os.remove(tmp)

    def test_android_builtin_fp32_model_bundled(self):
        """v1.19.1 Android 内置模型随 APK 打包（assets）：fp32 原版
        权重（v1.19.0 的 fp16 被 CPU EP Cast 内存炸弹否决：fp16 权重
        无原生 kernel、激活副本物化使内存达 fp32 的 ~6 倍，512×384
        即需 1.5GB、1024×768 直接 OOM SIGKILL，大图转换必闪退）。
        文件在仓库、体积完整、且**内部权重 dtype 必须为 fp32**——
        防止回归到 fp16。无权重依赖，CI 可跑（onnx 缺失时仅体积
        检查，dtype 断言跳过）。"""
        import os
        try:
            import onnx
        except ImportError:
            onnx = None
        path = os.path.join(os.path.dirname(__file__), "..",
                            "android", "app", "src", "main", "assets",
                            "models", "lineart.onnx")
        assert os.path.exists(path), f"内置模型缺失：{path}"
        size = os.path.getsize(path)
        assert 15 * 1024 * 1024 <= size <= 20 * 1024 * 1024, (
            f"内置 fp32 模型体积异常：{size/1048576:.2f}MB（预期约 16.4MB）")
        if onnx is None:
            return
        m = onnx.load(path)
        fp16 = [t.name for t in m.graph.initializer
                if t.data_type == onnx.TensorProto.FLOAT16]
        assert not fp16, (
            f"内置模型含 FLOAT16 权重 {fp16[:3]}...——fp16 在 CPU EP "
            "上触发 Cast 往返内存炸弹（v1.19.0 闪退根因），必须 fp32")

    @pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
    def test_builtin_model_identity_with_reference(self):
        """内置 fp32 模型与参考权重逐位一致（v1.19.1：内置即原版权重，
        同一文件应产出完全相同的 ink）。"""
        import os
        path = os.path.join(os.path.dirname(__file__), "..",
                            "android", "app", "src", "main", "assets",
                            "models", "lineart.onnx")
        if not os.path.exists(path):
            pytest.skip("内置模型不在仓库")
        bgr = np.full((200, 260, 3), 255, np.uint8)
        cv2.circle(bgr, (130, 100), 60, (60,) * 3, 8, cv2.LINE_AA)
        cv2.line(bgr, (40, 160), (220, 160), (50,) * 3, 3)
        ink_a = N.neural_semantic(bgr, path)
        ink_b = N.neural_semantic(bgr)
        assert np.array_equal(ink_a, ink_b), "内置模型与参考权重输出不一致"

    @pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
    def test_tiled_inference_precision(self):
        """v1.19.1 分块推理精度契约：超过内存预算（桌面 4GB 约 1600px）
        的图走 _infer_tiled（峰值 = 单块激活），与整图推理的 mask 层
        IoU ≥0.88（实测弱响应场景天花板，640 块 0.8810）。

        精度取舍：0.88 远好于 v1.19.0 必闪退，RELEASE_NOTES 如实记录。
        小图（内存预算内）走整图路径逐位不变。
        """
        from lineart.semantics.neural import (_infer_full, _infer_tiled,
                                               neural_postprocess)
        bgr = np.full((460, 520, 3), 235, np.uint8)
        cv2.circle(bgr, (260, 230), 150, (70,) * 3, 12, cv2.LINE_AA)
        cv2.rectangle(bgr, (60, 60), (200, 180), (80,) * 3, 5, cv2.LINE_AA)
        import onnxruntime as ort
        sess = ort.InferenceSession(N._find_model(),
                                    providers=["CPUExecutionProvider"])
        ref = np.clip(1.0 - _infer_full(sess, bgr), 0, 1).astype(np.float32)
        # 640 块：实测 IoU 0.8810（弱响应天花板）
        tiled = np.clip(1.0 - _infer_tiled(sess, bgr, tile=640), 0, 1).astype(np.float32)
        p = LineArtParams()
        mr = neural_postprocess(ref, p)
        mt = neural_postprocess(tiled, p)
        iou = (mr & mt).sum() / (mr | mt).sum()
        assert iou >= 0.88, f"分块推理 mask IoU 仅 {iou:.4f}（预期 ≥0.88）"

    @pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
    def test_neural_semantic_tiled_path_e2e(self):
        """管线端到端：大图（>内存预算）走 tiling 路径，输出墨线正常。

        使用 2048×1536 大图触发 tiling（>4GB 预算），
        检查墨线集中在预期半径附近（r=600±20px 内 ≥80% 墨线）。
        """
        # 2048×1536 大图触发 tiling 路径（>4GB 预算）
        bgr = np.full((1536, 2048, 3), 245, np.uint8)
        cv2.circle(bgr, (1024, 768), 600, (60,) * 3, 8, cv2.LINE_AA)
        out = to_lineart(bgr, LineArtParams(semantic="neural", mode="binary"))
        ink = out.max(axis=2) < 128 if out.ndim == 3 else out < 128
        assert ink.sum() > 5000, "tiling 路径墨线大量丢失"
        yy, xx = np.mgrid[0:1536, 0:2048]
        dist = np.sqrt((xx - 1024) ** 2 + (yy - 768) ** 2)
        # 检查墨线集中在 r=600 附近（±20px 内 ≥80% 墨线）
        ink_dist = dist[ink]
        near = np.abs(ink_dist - 600) <= 20
        assert near.mean() >= 0.80, (
            f"墨线位置偏移: 仅 {near.mean():.1%} 在 r=600±20px 内, "
            f"半径分布 p10/p50/p90={np.percentile(ink_dist,10):.0f}/"
            f"{np.percentile(ink_dist,50):.0f}/"
            f"{np.percentile(ink_dist,90):.0f}")
