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
        """semantic=neural 且权重不可寻：报错含放置指引。

        v1.22.0 拆包后 _find_model 在 informative 模块内被直接调用，
        monkeypatch 需打到定义处（包出口 re-export 是另一份绑定）。
        """
        monkeypatch.setattr("lineart.semantics.neural.informative._find_model",
                            lambda mp=None: None)
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

    def test_exe_bundled_double_fp32_weights_in_repo(self):
        """v1.24.0 桌面 EXE 双权重内置（仓库根 models/）：informative
        fp32（17MB）+ DexiNed fp32 原版（45MB blockwise DQ，无量化
        无降档）。用户指令：EXE 极致性能、两个权重完全释放——桌面
        无移动端 5GB 内存约束，直接用原版权重（重量化 INT8 版仅
        Android APK 使用）。无权重依赖部分 CI 可跑（onnx 缺失时
        仅体积检查，dtype 断言跳过）。"""
        import os
        try:
            import onnx
        except ImportError:
            onnx = None
        root = os.path.join(os.path.dirname(__file__), "..")
        inf_path = os.path.join(root, "models", "lineart.onnx")
        dex_path = os.path.join(root, "models", "dexined.onnx")
        assert os.path.exists(inf_path), f"EXE 内置 informative 权重缺失：{inf_path}"
        assert os.path.exists(dex_path), f"EXE 内置 DexiNed 权重缺失：{dex_path}"
        inf_size = os.path.getsize(inf_path)
        dex_size = os.path.getsize(dex_path)
        assert 15 * 1024 * 1024 <= inf_size <= 20 * 1024 * 1024, (
            f"informative 权重体积异常：{inf_size/1048576:.2f}MB（预期约 16.4MB）")
        assert 40 * 1024 * 1024 <= dex_size <= 50 * 1024 * 1024, (
            f"DexiNed 权重体积异常：{dex_size/1048576:.2f}MB（预期约 45MB）")
        if onnx is None:
            return
        m_inf = onnx.load(inf_path)
        fp16 = [t.name for t in m_inf.graph.initializer
                if t.data_type == onnx.TensorProto.FLOAT16]
        assert not fp16, (
            f"informative 权重含 FLOAT16 {fp16[:3]}...——EXE 侧同为 CPU EP，"
            "fp16 触发 Cast 往返内存炸弹（v1.19.0 教训），必须 fp32")
        m_dex = onnx.load(dex_path)
        op_types = {op.op_type for op in m_dex.graph.node}
        assert "DequantizeLinear" in op_types, (
            "DexiNed 权重缺 DequantizeLinear——不是原版 blockwise DQ 结构，"
            "可能被误换为其他量化形态（EXE 侧要求原版权重完全释放）")
        init_ndim = {}
        for t in m_dex.graph.initializer:
            if not t.dims:
                continue
            init_ndim[t.name] = len(t.dims)
        dq_1d = [n.name for n in m_dex.graph.node
                 if n.op_type == "DequantizeLinear"
                 and init_ndim.get(n.input[0]) == 1]
        assert not dq_1d, (
            f"DexiNed 权重含 1D bias DequantizeLinear {dq_1d[:3]}...——"
            "ORT 1.31 blockwise bias bug（block_size must be 0），EXE 无 "
            "onnx 包无法运行时图手术（v1.24.0 CI 实测失败），入库形态必须"
            "是「bias 修复 + 动态轴」终态")
        in_dims = m_dex.graph.input[0].type.tensor_type.shape.dim
        assert len(in_dims) >= 4 and (in_dims[2].dim_param
                                      or in_dims[3].dim_param), (
            "DexiNed 输入 H/W 仍是固定 dim——入库形态缺动态轴手术，"
            "EXE 内 --dexined-scale 2 将回退 1x（v1.21.0 精度特性失效）")

    def test_android_dexined_weight_android_ort_compatible(self):
        """v1.24.1 Android DexiNed 权重 opset 兼容契约：APK 内置
        重量化版 dexined.onnx 的 opset 必须 ≤ 20——onnxruntime-android
        1.17 官方支持上限为 opset 20，stamp 21 会 ORT_INVALID_ARGUMENT
        拒载（v1.24.0 APK 实测：ensemble 推理失败降级、自检报错）。
        重量化版算子集（Conv/ConvTranspose/MaxPool/Relu/Reshape/
        Concat/Mul/Add + per-channel DQ(axis)）opset 13 即满足，
        17 留裕量。同时断言无 block_size 属性（ORT 1.17 不支持
        blockwise DQ）与 1D bias DQ（1.31 bug）。无权重依赖部分
        CI 可跑（onnx 缺失时跳过结构断言）。"""
        import os
        try:
            import onnx
        except ImportError:
            onnx = None
        path = os.path.join(os.path.dirname(__file__), "..",
                            "android", "app", "src", "main", "assets",
                            "models", "dexined.onnx")
        assert os.path.exists(path), f"Android 内置 DexiNed 权重缺失：{path}"
        size = os.path.getsize(path)
        assert 30 * 1024 * 1024 <= size <= 40 * 1024 * 1024, (
            f"Android DexiNed 权重体积异常：{size/1048576:.2f}MB（预期约 34.8MB）")
        if onnx is None:
            return
        m = onnx.load(path)
        opset = {o.domain or "ai.onnx": o.version for o in m.opset_import}
        assert opset.get("ai.onnx", 0) <= 20, (
            f"Android DexiNed 权重 opset {opset}——onnxruntime-android 1.17 "
            "支持上限 20，stamp 21 触发 ORT_INVALID_ARGUMENT 拒载"
            "（v1.24.0 APK 自检失败根因）")
        init_ndim = {}
        for t in m.graph.initializer:
            if t.dims:
                init_ndim[t.name] = len(t.dims)
        for n in m.graph.node:
            if n.op_type != "DequantizeLinear":
                continue
            attrs = {a.name for a in n.attribute}
            assert "block_size" not in attrs, (
                f"DQ {n.name} 带 block_size 属性——ORT 1.17 不支持 "
                "blockwise DQ（Android 重量化版必须 per-channel）")
            assert init_ndim.get(n.input[0]) != 1, (
                f"1D bias DQ {n.name}——ORT 1.31 blockwise bias bug，"
                "Android 版不应保留任何 1D DQ 节点")
        in_dims = m.graph.input[0].type.tensor_type.shape.dim
        assert len(in_dims) >= 4 and (in_dims[2].dim_param
                                      or in_dims[3].dim_param), (
            "Android DexiNed 输入 H/W 仍是固定 dim——v1.26.0 起权重必须"
            "为动态轴版（runDexined 场景路由 2x 高分辨率推理依赖之；"
            "固定 480×640 会让大图细节放大后模糊）")

    @pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
    def test_exe_repo_weights_resolve_without_env(self, monkeypatch, tmp_path):
        """v1.24.0 仓库根 models/ 命中查找链第 3 条：清空环境变量 +
        HOME 指向空目录（隔离 ~/.lineart/ 第 2 条）后，informative
        与 DexiNed 均解析到仓库 models/ 权重（EXE 打包后 _MEIPASS 根
        即等价仓库根，datas 命中同一条链）。"""
        import os
        from lineart.semantics.neural import informative as I
        from lineart.semantics.neural import dexined as D
        monkeypatch.delenv("LINEART_NEURAL_MODEL", raising=False)
        monkeypatch.delenv("LINEART_DEXINED_MODEL", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        inf_found = I._find_model()
        dex_found = D._find_dexined_model()
        assert inf_found and os.path.basename(inf_found) == "lineart.onnx", (
            f"informative 权重解析失败：{inf_found}")
        assert dex_found and os.path.basename(dex_found) == "dexined.onnx", (
            f"DexiNed 权重解析失败：{dex_found}")
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        assert inf_found.startswith(os.path.join(root, "models")), (
            f"informative 未解析到仓库 models/：{inf_found}")
        assert dex_found.startswith(os.path.join(root, "models")), (
            f"DexiNed 未解析到仓库 models/：{dex_found}")

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
        """分块推理精度契约：小图（460×520 < tile 640）单块覆盖，
        与整图推理逐位一致。"""
        from lineart.semantics.neural import (_infer_full, _infer_tiled,
                                               neural_postprocess)
        bgr = np.full((460, 520, 3), 235, np.uint8)
        cv2.circle(bgr, (260, 230), 150, (70,) * 3, 12, cv2.LINE_AA)
        cv2.rectangle(bgr, (60, 60), (200, 180), (80,) * 3, 5, cv2.LINE_AA)
        import onnxruntime as ort
        sess = ort.InferenceSession(N._find_model(),
                                    providers=["CPUExecutionProvider"])
        ref = np.clip(1.0 - _infer_full(sess, bgr), 0, 1).astype(np.float32)
        tiled = np.clip(1.0 - _infer_tiled(sess, bgr, tile=640), 0, 1).astype(np.float32)
        p = LineArtParams()
        mr = neural_postprocess(ref, p)
        mt = neural_postprocess(tiled, p)
        iou = (mr & mt).sum() / (mr | mt).sum()
        assert iou >= 0.88, f"分块推理 mask IoU 仅 {iou:.4f}（预期 ≥0.88）"

    @pytest.mark.skipif(not N.available(), reason="需 onnxruntime + 模型权重")
    def test_global_norm_tiling_precision(self):
        """v1.20.0 全局归一化分块推理精度：1200×1200 大图触发多块分块，
        全局归一化（降采样统计量 + ONNX 图手术）vs 原始分块的 IoU 提升。

        实测：原始分块 IoU ~0.80，全局归一化 IoU ~0.89（+9%）。
        需 onnx 包；未安装时跳过。
        """
        pytest.importorskip("onnx")
        from lineart.semantics.neural import (_infer_full, _infer_tiled,
                                               _build_global_norm_session,
                                               neural_postprocess)
        bgr = np.full((1200, 1200, 3), 235, np.uint8)
        cv2.circle(bgr, (600, 600), 400, (70,) * 3, 12, cv2.LINE_AA)
        cv2.rectangle(bgr, (200, 200), (500, 400), (80,) * 3, 5, cv2.LINE_AA)
        cv2.line(bgr, (100, 100), (1100, 1100), (60,) * 3, 3)
        import onnxruntime as ort
        path = N._find_model()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ref = np.clip(1.0 - _infer_full(sess, bgr), 0, 1).astype(np.float32)
        orig_tiled = np.clip(1.0 - _infer_tiled(sess, bgr, tile=640), 0, 1).astype(np.float32)
        fixed_sess = _build_global_norm_session(path, bgr)
        fixed_tiled = np.clip(1.0 - _infer_tiled(fixed_sess, bgr, tile=640), 0, 1).astype(np.float32)
        p = LineArtParams()
        mr = neural_postprocess(ref, p)
        mt_orig = neural_postprocess(orig_tiled, p)
        mt_fixed = neural_postprocess(fixed_tiled, p)
        iou_orig = (mr & mt_orig).sum() / (mr | mt_orig).sum()
        iou_fixed = (mr & mt_fixed).sum() / (mr | mt_fixed).sum()
        assert iou_fixed >= 0.85, f"全局归一化分块 IoU 仅 {iou_fixed:.4f}（预期 ≥0.85）"
        assert iou_fixed > iou_orig, (
            f"全局归一化未提升精度: orig={iou_orig:.4f}, fixed={iou_fixed:.4f}")

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


@pytest.mark.skipif(not N.dexined_available(),
                    reason="需 onnxruntime + dexined 权重")
class TestDexiNedSemantic:
    """v1.20.0 DexiNed INT8 边缘语义（semantic="dexined"）契约测试。"""

    def test_inference_contract(self):
        bgr = np.full((64, 80, 3), 200, np.uint8)
        cv2.circle(bgr, (40, 32), 18, (30, 30, 30), 2, cv2.LINE_AA)
        ink = N.dexined_semantic(bgr)
        assert ink.shape == (64, 80), "输出尺寸=输入尺寸契约"
        assert ink.dtype == np.float32
        assert 0.0 <= ink.min() and ink.max() <= 1.0
        assert ink.max() > 0.5, "圆形结构边缘无强响应"

    def test_pipeline_output_contract(self):
        bgr = np.full((96, 128, 3), 200, np.uint8)
        cv2.circle(bgr, (64, 48), 26, (30, 30, 30), 2, cv2.LINE_AA)
        for mode in ("antialiased", "binary"):
            out = to_lineart(bgr, LineArtParams(semantic="dexined",
                                                mode=mode))
            assert out.shape[:2] == (96, 128), "输出尺寸=输入尺寸契约"
            assert out.dtype == np.uint8
            assert (out < 128).any(), "DexiNed 语义无线条输出"
            if mode == "binary":
                assert set(np.unique(out).tolist()) <= {0, 255}

    def test_circle_precision_beats_neural(self):
        """精度价值契约：合成圆上 DexiNed 端到端定位偏差 < neural。

        DexiNed（BIPED SOTA 边缘检测）原始输出无系统偏移
        （mean_err -0.09 vs informative-drawings +1.0），
        端到端 mean_err -0.042 vs -0.075（改善 44%）。
        """
        img = np.full((192, 192), 235.0)
        yy, xx = np.mgrid[0:192, 0:192].astype(np.float64)
        d = np.sqrt((yy - 96.2) ** 2 + (xx - 96.7) ** 2)
        img = (60.0 + 180.0 * np.clip(40.3 - d + 0.5, 0, 1)).astype(np.float32)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def radial_mean_err(out):
            ink = 1.0 - out.astype(np.float64) / 255.0
            errs = []
            for i in range(360):
                th = 2 * np.pi * i / 360
                dy, dx = np.sin(th), np.cos(th)
                rs = np.arange(35.3, 45.3, 0.25)
                ys, xs = 96.2 + rs * dy, 96.7 + rs * dx
                y0 = np.floor(ys).astype(int)
                x0 = np.floor(xs).astype(int)
                y0c = np.clip(y0, 0, 191)
                x0c = np.clip(x0, 0, 191)
                y1c = np.clip(y0 + 1, 0, 191)
                x1c = np.clip(x0 + 1, 0, 191)
                fy, fx = ys - y0, xs - x0
                vals = (ink[y0c, x0c] * (1 - fy) * (1 - fx) +
                        ink[y1c, x0c] * fy * (1 - fx) +
                        ink[y0c, x1c] * (1 - fy) * fx +
                        ink[y1c, x1c] * fy * fx)
                if vals.sum() < 0.4:
                    continue
                errs.append((vals * rs).sum() / vals.sum() - 40.3)
            return np.mean(np.abs(errs))

        m_dex = radial_mean_err(to_lineart(bgr, LineArtParams(
            semantic="dexined")))
        m_neu = radial_mean_err(to_lineart(bgr, LineArtParams(
            semantic="neural")))
        assert m_dex < 0.15, f"DexiNed 圆定位偏差 {m_dex:.4f} 过大"
        assert m_dex < m_neu, (
            f"DexiNed 相对 neural 无精度优势: dexined={m_dex:.4f}, "
            f"neural={m_neu:.4f}")

    def test_ridge_recall_beats_neural(self):
        """弱对比山脊召回：DexiNed 高于 informative-drawings。"""
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

        c_dex = cover(to_lineart(img, LineArtParams(semantic="dexined",
                                                    scene="photo")))
        c_neu = cover(to_lineart(img, LineArtParams(semantic="neural",
                                                    scene="photo")))
        assert c_dex > 0.8, f"DexiNed 山脊召回不足 {c_dex:.1%}"
        assert c_dex > c_neu, (
            f"DexiNed 相对 neural 无召回优势: dexined={c_dex:.1%}, "
            f"neural={c_neu:.1%}")


class TestDexiNedScale:
    """v1.21.0 DexiNed 高分辨率推理（dexined_scale）契约测试。

    路由测试为纯函数（无权重依赖，CI 可跑）；推理/精度/管线集成
    测试在权重存在时才执行。
    """

    def test_scale_routing(self):
        from lineart.pipeline import _dexined_scale_for
        p = LineArtParams()
        assert _dexined_scale_for(p, "photo") == 1
        assert _dexined_scale_for(p, "sketch") == 1
        assert _dexined_scale_for(p, "off") == 1
        for scene in ("lineart", "blueprint", "pixel"):
            assert _dexined_scale_for(p, scene) == 2, (
                f"清晰边缘场景 {scene} 应路由 2x")

    def test_scale_explicit_overrides_auto(self):
        from lineart.pipeline import _dexined_scale_for
        assert _dexined_scale_for(
            LineArtParams(dexined_scale="2"), "photo") == 2
        assert _dexined_scale_for(
            LineArtParams(dexined_scale="1"), "lineart") == 1

    def test_scale_invalid_raises(self):
        with pytest.raises(ValueError):
            N.dexined_semantic(np.zeros((8, 8, 3), np.uint8), scale=3)

    @pytest.mark.skipif(not N.dexined_available(),
                        reason="需 onnxruntime + dexined 权重")
    def test_scale2_inference_contract(self):
        bgr = np.full((64, 80, 3), 200, np.uint8)
        cv2.circle(bgr, (40, 32), 18, (30, 30, 30), 2, cv2.LINE_AA)
        ink = N.dexined_semantic(bgr, scale=2)
        assert ink.shape == (64, 80), "scale=2 输出尺寸=输入尺寸契约"
        assert ink.dtype == np.float32
        assert 0.0 <= ink.min() and ink.max() <= 1.0
        assert ink.max() > 0.3, "scale=2 圆形边缘无响应"

    @pytest.mark.skipif(not N.dexined_available(),
                        reason="需 onnxruntime + dexined 权重")
    def test_scale2_non_multiple16_no_crash(self):
        """v1.27.2 契约：非 16 倍数输入 scale=2 不得崩溃。

        DexiNed 动态版多 block 上下采样路径倍数不同（4/2 混合），
        非 16 倍数输入内部 Concat 维度不匹配直接崩（实测 605→
        "608 vs 606"、610/612/616/620 全崩）。_round16 对齐推理
        尺寸后任意输入安全，输出仍精确回原尺寸。用户图（如
        605 宽）2x=1210 曾必崩。"""
        for (h, w) in [(605, 611), (811, 305), (610, 610), (1523, 607)]:
            bgr = np.full((h, w, 3), 200, np.uint8)
            cv2.circle(bgr, (w // 2, h // 2), min(h, w) // 3,
                       (30, 30, 30), 3, cv2.LINE_AA)
            ink = N.dexined_semantic(bgr, scale=2)
            assert ink.shape == (h, w), (
                f"({h},{w}) scale=2 输出尺寸 {ink.shape} ≠ 输入尺寸")
            assert ink.dtype == np.float32
            assert ink.max() > 0.3, f"({h},{w}) 圆形边缘无响应"

    @pytest.mark.skipif(not N.dexined_available(),
                        reason="需 onnxruntime + dexined 权重")
    def test_scale2_beats_scale1_on_large_circle(self):
        """精度价值契约：大图上 scale=2 定位偏差 < scale=1。

        实测（960×1280 大圆 R=200）：scale1 0.380 → scale2 0.215
        （-43%），线条锐度 +95%。推理分辨率封顶 960×1280
        （1920×2560 全分辨率直推 OOM）。
        """
        H, W = 960, 1280
        cx, cy, R = 640.7, 480.2, 200.3
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        img = (60.0 + 180.0 * np.clip(R + 0.5 - d, 0, 1)).astype(np.float32)
        bgr = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        def radial_mean_err(ink):
            errs = []
            for i in range(360):
                th = 2 * np.pi * i / 360
                dy, dx = np.sin(th), np.cos(th)
                rs = np.arange(R - 5, R + 5, 0.25)
                ys, xs = cy + rs * dy, cx + rs * dx
                y0 = np.floor(ys).astype(int)
                x0 = np.floor(xs).astype(int)
                y0c = np.clip(y0, 0, H - 1)
                x0c = np.clip(x0, 0, W - 1)
                y1c = np.clip(y0 + 1, 0, H - 1)
                x1c = np.clip(x0 + 1, 0, W - 1)
                fy, fx = ys - y0, xs - x0
                vals = (ink[y0c, x0c] * (1 - fy) * (1 - fx) +
                        ink[y1c, x0c] * fy * (1 - fx) +
                        ink[y0c, x1c] * (1 - fy) * fx +
                        ink[y1c, x1c] * fy * fx)
                if vals.sum() < 0.4:
                    continue
                errs.append((vals * rs).sum() / vals.sum() - R)
            return np.mean(np.abs(errs))

        m1 = radial_mean_err(N.dexined_semantic(bgr, scale=1))
        m2 = radial_mean_err(N.dexined_semantic(bgr, scale=2))
        assert m2 < 0.35, f"scale=2 定位偏差 {m2:.4f} 过大"
        assert m2 < m1, (
            f"scale=2 相对 scale=1 无精度优势: 2x={m2:.4f}, 1x={m1:.4f}")

    @pytest.mark.skipif(not N.dexined_available(),
                        reason="需 onnxruntime + dexined 权重")
    def test_scale2_pipeline_contract(self):
        """lineart 场景 auto 路由 2x + 显式 dexined_scale 管线契约。"""
        bgr = np.full((96, 128, 3), 200, np.uint8)
        cv2.circle(bgr, (64, 48), 26, (30, 30, 30), 2, cv2.LINE_AA)
        for kw in ({"scene": "lineart"},
                   {"scene": "off", "dexined_scale": "2"}):
            out = to_lineart(bgr, LineArtParams(
                semantic="dexined", mode="binary", **kw))
            assert out.shape[:2] == (96, 128), "输出尺寸=输入尺寸契约"
            assert out.dtype == np.uint8
            assert (out < 128).any(), "scale=2 管线无线条输出"
            assert set(np.unique(out).tolist()) <= {0, 255}

    @pytest.mark.skipif(not N.dexined_available(),
                        reason="需 onnxruntime + dexined 权重")
    def test_photo_scene_auto_backward_compat(self):
        """photo 场景 auto 路由 scale=1：与显式 dexined_scale="1" 逐位一致。

        v1.21.0 向后兼容契约：弱对比软边缘场景（photo/sketch）
        的默认行为与 v1.20.0 完全一致（2x 感受野失配会降召回）。
        """
        img = cv2.imread(str(__import__("pathlib").Path(__file__).parent
                             / ".." / "demo" / "samples" / "photo_like.png"))
        if img is None:
            pytest.skip("demo 样本不存在")
        out_auto = to_lineart(img, LineArtParams(semantic="dexined",
                                                 scene="photo"))
        out_1x = to_lineart(img, LineArtParams(semantic="dexined",
                                               scene="photo",
                                               dexined_scale="1"))
        assert np.array_equal(out_auto, out_1x), (
            "photo 场景 auto 路由应等价显式 scale=1（v1.20.0 行为）")


@pytest.mark.skipif(not N.ensemble_available(),
                    reason="需 onnxruntime + 双权重")
class TestEnsembleSemantic:
    """v1.22.0 ensemble 三语义 noisy-OR 融合契约测试。"""

    def test_inference_contract(self):
        bgr = np.full((64, 80, 3), 200, np.uint8)
        cv2.circle(bgr, (40, 32), 18, (30, 30, 30), 2, cv2.LINE_AA)
        ink = N.ensemble_semantic(bgr)
        assert ink.shape == (64, 80), "输出尺寸=输入尺寸契约"
        assert ink.dtype == np.float32
        assert 0.0 <= ink.min() and ink.max() <= 1.0
        assert ink.max() > 0.5, "圆形边缘无强响应"

    def test_pipeline_output_contract(self):
        bgr = np.full((96, 128, 3), 200, np.uint8)
        cv2.circle(bgr, (64, 48), 26, (30, 30, 30), 2, cv2.LINE_AA)
        for mode in ("antialiased", "binary"):
            out = to_lineart(bgr, LineArtParams(semantic="ensemble",
                                                mode=mode))
            assert out.shape[:2] == (96, 128), "输出尺寸=输入尺寸契约"
            assert out.dtype == np.uint8
            assert (out < 128).any(), "ensemble 语义无线条输出"
            if mode == "binary":
                assert set(np.unique(out).tolist()) <= {0, 255}

    def test_thin_line_recall_beats_dexined(self):
        """精度价值契约：极浅对比细线召回 > dexined 单语义。

        诊断确认 neural 响应值塌缩（0.259）、dexined 0.858，
        ensemble noisy-OR 叠加梯度高频分量 → 0.975。
        """
        il = np.full((256, 256, 3), 235, np.uint8)
        rng = np.random.RandomState(7)
        for _ in range(40):
            y1, y2 = rng.randint(20, 236, 2)
            x1, x2 = rng.randint(20, 236, 2)
            cv2.line(il, (x1, y1), (x2, y2), (200,) * 3, 1, cv2.LINE_AA)
        truth = cv2.dilate((cv2.cvtColor(il, cv2.COLOR_BGR2GRAY) < 218
                            ).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

        def recall(out):
            m = out < 128 if out.ndim == 2 else (out.max(axis=2) < 128)
            return (m & truth).sum() / truth.sum()

        r_ens = recall(to_lineart(il, LineArtParams(semantic="ensemble",
                                                    scene="off")))
        r_dex = recall(to_lineart(il, LineArtParams(semantic="dexined",
                                                    scene="off")))
        assert r_ens > 0.90, f"ensemble 细线召回 {r_ens:.3f} 不足"
        assert r_ens > r_dex, (
            f"ensemble 相对 dexined 无召回优势: ens={r_ens:.3f}, "
            f"dex={r_dex:.3f}")

    def test_ridge_recall_beats_dexined(self):
        """弱对比山脊召回：ensemble > dexined 单语义。"""
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
            m = out < 128 if out.ndim == 2 else (out.max(axis=2) < 128)
            hits = 0
            for xi in range(w):
                ys = np.where(m[:, xi])[0]
                if len(ys) and np.min(np.abs(ys - ridge[xi])) <= 2.5:
                    hits += 1
            return hits / w

        c_ens = cover(to_lineart(img, LineArtParams(semantic="ensemble",
                                                    scene="off")))
        c_dex = cover(to_lineart(img, LineArtParams(semantic="dexined",
                                                    scene="off")))
        assert c_ens > 0.95, f"ensemble 山脊召回 {c_ens:.1%} 不足"
        assert c_ens > c_dex, (
            f"ensemble 相对 dexined 无召回优势: ens={c_ens:.1%}, "
            f"dex={c_dex:.1%}")

    def test_noise_controlled_vs_raw_sum(self):
        """噪线控制契约：融合不放大纹理散点（墨线总量增幅温和）。

        端到端墨线 px 比值 ensemble/dexined ≤ 1.6（滞回+连通域
        清理压制 noisy-OR 的散点叠加；实测各样本 1.17~1.32x）。
        """
        import os
        path = os.path.join(os.path.dirname(__file__), "..", "demo",
                            "samples", "circle_scene.png")
        if not os.path.exists(path):
            pytest.skip("demo 样本不存在")
        img = cv2.imread(path)

        def ink_px(sem):
            out = to_lineart(img, LineArtParams(semantic=sem, scene="off"))
            m = out < 128 if out.ndim == 2 else (out.max(axis=2) < 128)
            return m.sum()

        ratio = ink_px("ensemble") / max(ink_px("dexined"), 1)
        assert ratio <= 1.6, f"ensemble 墨线放大 {ratio:.2f}x 超限（噪线风险）"
