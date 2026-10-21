"""informative-drawings 线稿语义（semantic="neural"，v1.17.0 内置）。

carolineec/informative-drawings (CVPR'22) 无配对线稿生成，
ONNX 转换版权重 rocca/informative-drawings-line-art-onnx（17MB）。

推理架构：
- 小图（内存预算内，桌面 4GB 约 1600px 长边）→ 整图前向 = 逐位一致
- 大图 → 全局归一化分块推理（v1.20.0）：降采样图估计 InstanceNorm
  全局统计量 → ONNX 图手术替换为固定仿射变换 → 分块推理，
  消除分块局部统计偏差（IoU 0.80→0.89，1200² 弱响应合成实测）；
  需 onnx 包，未安装时回退原始分块（IoU 0.80，仍可用）。
- fp32 激活内存实测 ~1.55KB/px（2048×1536 整图需 4.7GB），
  v1.19.1 起分块混合派发。

v1.22.0 细节缺失诊断定位（见 research_report）：本模型对弱/浅对比
细节的 ink 响应值塌缩——原始输出集中 <0.5，被 0.5 阈值截断
（阈值 0.3→0.5 时细线召回 0.41→0.27 陡降；vs DexiNed 仅降 0.10）。
该短板由 ensemble 融合（ensemble.py）的 noisy-OR 概率叠加补偿。

供应链约定：onnxruntime 与 17MB 权重不进核心安装，用户自行放置
权重后显式启用（semantic="neural"）。本模块不自动下载权重。
"""
import os
from pathlib import Path

import cv2
import numpy as np

_MODEL_NAME = "lineart.onnx"
_MODEL_SOURCE = ("https://hf-mirror.com/rocca/"
                 "informative-drawings-line-art-onnx (resolve/main/lineart.onnx)")

_SESSION = {"path": None, "sess": None}
_STATS_SESSION = {"path": None, "sess": None}

# v1.19.1 分块推理内存方案（详见模块 docstring）
_TILE_SMALL = 384
_TILE_LARGE = 640
_OVERLAP = 64
_MEMORY_BUDGET = 4 * 1024 * 1024 * 1024  # 4GB 桌面预算
_BYTES_PER_PX = 1550  # fp32 激活实测 ~1.55KB/px
_DS_SIZE = 896  # 全局统计量降采样尺寸（v1.22.0: 640→896 时间换精度，同图实测 IoU 0.78→0.86/+8.2pp；960/1100 更高但统计推理峰值逼近 4GB 预算，移动端与多 session 测试进程会 OOM——OOM 是崩溃约束非时间，故取 896 平衡档）


def _candidate_paths(model_path=None):
    if model_path:
        return [Path(model_path)]
    paths = []
    env = os.environ.get("LINEART_NEURAL_MODEL")
    if env:
        paths.append(Path(env))
    paths.append(Path.home() / ".lineart" / _MODEL_NAME)
    paths.append(Path(__file__).resolve().parent.parent.parent.parent
                 / "models" / _MODEL_NAME)
    return paths


def _find_model(model_path=None):
    for p in _candidate_paths(model_path):
        if p.is_file():
            return str(p)
    return None


def _try_ort():
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def available():
    """神经语义是否可用（onnxruntime 已装 + 权重可寻）。"""
    return _try_ort() and _find_model() is not None


def _infer_full(sess, bgr):
    """整图推理（BGR uint8 → 输出 y，1=背景 0=线）。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]
    return sess.run(["output"], {"input": x})[0][0, 0]


def _infer_tiled(sess, bgr, tile=_TILE_SMALL, overlap=_OVERLAP):
    """分块推理：重叠块中心加权混合（v1.19.1 内存方案）。

    每块输出按「距块边缘的线性坡」加权累加（坡长 overlap//2+1），
    块边界上下文差异被混合稀释；块贴边收缩保证覆盖全图。
    精度：tile=640 时 mask IoU 0.8810（弱响应场景天花板），
    tile=384 时 0.7986。大 tile 精度更高但内存峰值更大。
    """
    h, w = bgr.shape[:2]
    acc = np.zeros((h, w), np.float32)
    wacc = np.zeros((h, w), np.float32)
    stride = tile - overlap
    ys = list(range(0, max(h - tile, 0) + 1, stride))
    xs = list(range(0, max(w - tile, 0) + 1, stride))
    if ys[-1] + tile < h:
        ys.append(h - tile)
    if xs[-1] + tile < w:
        xs.append(w - tile)
    edge = overlap // 2 + 1
    for y0 in ys:
        for x0 in xs:
            y1, x1 = min(y0 + tile, h), min(x0 + tile, w)
            y0c, x0c = max(y1 - tile, 0), max(x1 - tile, 0)
            y = _infer_full(sess, bgr[y0c:y1, x0c:x1])
            bh, bw = y.shape
            wy = np.minimum(np.arange(bh) + 1, bh - np.arange(bh))
            wx = np.minimum(np.arange(bw) + 1, bw - np.arange(bw))
            wgt = np.outer(np.minimum(wy, edge),
                           np.minimum(wx, edge)).astype(np.float32)
            acc[y0c:y1, x0c:x1] += y * wgt
            wacc[y0c:y1, x0c:x1] += wgt
    return acc / np.maximum(wacc, 1e-6)


def _get_stats_session(model_path):
    """获取或创建缓存的统计量提取模型（按模型路径缓存）。

    在每个 InstanceNormalization 输入前插入 Identity 节点作为额外输出，
    用于提取各层的激活统计量（per-channel mean/var）。
    """
    if _STATS_SESSION["path"] != model_path:
        import onnx
        from onnx import helper, TensorProto
        import onnxruntime as ort
        model = onnx.load(model_path)
        inorm_nodes = [n for n in model.graph.node
                       if n.op_type == "InstanceNormalization"]
        for idx, node in enumerate(inorm_nodes):
            sn = f"stats_{idx}"
            id_node = helper.make_node("Identity", [node.input[0]], [sn],
                                       name=f"StatsId_{idx}")
            for j, n in enumerate(model.graph.node):
                if n.name == node.name:
                    model.graph.node.insert(j, id_node)
                    break
            model.graph.output.append(
                helper.make_tensor_value_info(sn, TensorProto.FLOAT,
                                              [1, "C", "H", "W"]))
        _STATS_SESSION["sess"] = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"])
        _STATS_SESSION["path"] = model_path
    return _STATS_SESSION["sess"]


def _build_global_norm_session(model_path, bgr, ds_size=_DS_SIZE):
    """用降采样图估计全局 InstanceNorm 统计量，创建固定归一化模型。

    ONNX 图手术：将每个 InstanceNormalization 替换为 Mul+Add 固定仿射变换，
    使用降采样图提取的 per-channel 全局 mean/var。消除分块推理中
    InstanceNorm 块内统计偏差（IoU 0.80→0.89，1200² 弱响应合成实测）。

    开销：一次 ds_size×ds_size 推理（~0.1s）+ InferenceSession 创建（~0.5s）。
    """
    import onnx
    from onnx import helper, numpy_helper, TensorProto
    import onnxruntime as ort

    h, w = bgr.shape[:2]
    if min(h, w) <= ds_size:
        bgr_ds = bgr
    else:
        scale = ds_size / max(h, w)
        bgr_ds = cv2.resize(bgr, (max(1, int(w * scale)),
                                  max(1, int(h * scale))),
                            interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr_ds, cv2.COLOR_BGR2RGB)
    x = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None]

    stats_sess = _get_stats_session(model_path)
    results = stats_sess.run(None, {"input": x})
    global_stats = [(a.mean(axis=(2, 3))[0], a.var(axis=(2, 3))[0])
                    for a in results[1:]]

    model = onnx.load(model_path)
    inorm_nodes = [n for n in model.graph.node
                   if n.op_type == "InstanceNormalization"]
    eps = 1e-5
    for i, node in enumerate(inorm_nodes):
        mean, var = global_stats[i]
        scale = (1.0 / np.sqrt(var + eps)).astype(np.float32).reshape(1, -1, 1, 1)
        shift = (-mean.astype(np.float32).reshape(1, -1, 1, 1) * scale)
        sn, shn = f"gs_{i}", f"gh_{i}"
        model.graph.initializer.append(numpy_helper.from_array(scale, sn))
        model.graph.initializer.append(numpy_helper.from_array(shift, shn))
        mn = f"gm_{i}"
        mul_n = helper.make_node("Mul", [node.input[0], sn], [mn], name=f"GNM_{i}")
        add_n = helper.make_node("Add", [mn, shn], [node.output[0]], name=f"GNA_{i}")
        for j, n in enumerate(model.graph.node):
            if n.name == node.name:
                model.graph.node.pop(j)
                model.graph.node.insert(j, mul_n)
                model.graph.node.insert(j + 1, add_n)
                break

    return ort.InferenceSession(model.SerializeToString(),
                                providers=["CPUExecutionProvider"])


def _memory_budget_tile(h, w, budget=_MEMORY_BUDGET):
    """按内存预算计算最大可承受 tile 尺寸（fp32 ~1.55KB/px）。

    返回 tile 尺寸，若整图可承受则返回 None（走整图路径）。
    """
    full_mem = h * w * _BYTES_PER_PX
    if full_mem <= budget:
        return None  # 整图可承受
    # 单块激活 ≈ tile² * _BYTES_PER_PX，取预算内最大 tile
    tile = int(np.sqrt(budget / _BYTES_PER_PX))
    # 对齐到 64 的倍数（与 overlap 兼容）
    tile = max(_TILE_SMALL, (tile // 64) * 64)
    return min(tile, max(h, w))


def neural_semantic(img_bgr, model_path=None):
    """informative-drawings lineart ONNX 推理 → ink 语义图（[0,1]，1=线）。

    输入 BGR/灰度（uint8 或 float [0,1]），内部转 NCHW RGB [0,1]；
    模型输出 1=背景、0=线，反相统一为管线 ink 约定（1=线）。
    Session 按权重路径缓存（重复调用零开销）。
    混合派发：
      - 小图（内存预算内，桌面 4GB 约 1600px 长边）→ 整图推理 = 逐位一致
      - 大图 → 全局归一化分块推理（v1.20.0）
    v1.20.0 全局归一化：降采样图估计 InstanceNorm 全局统计量 → ONNX 图手术
    替换为固定仿射变换 → 分块推理。消除 InstanceNorm 块内统计偏差
    （IoU 0.80→0.89，1200² 弱响应合成实测）。需 onnx 包；未安装时回退原始分块。
    """
    import onnxruntime as ort
    path = _find_model(model_path)
    if path is None:
        raise FileNotFoundError(
            "未找到神经语义模型权重（semantic=\"neural\"）。请将 "
            f"informative-drawings lineart ONNX（约 17MB，文件名 {_MODEL_NAME}）"
            "放置于以下任一位置：① 环境变量 LINEART_NEURAL_MODEL 指定的路径；"
            "② ~/.lineart/lineart.onnx；③ 仓库根目录 models/lineart.onnx；"
            "或通过 params.neural_model / CLI --neural-model 显式指定。"
            f"权重来源：{_MODEL_SOURCE}")
    if _SESSION["path"] != path:
        _SESSION["sess"] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        _SESSION["path"] = path
    bgr = np.asarray(img_bgr)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr * 255.0 if bgr.max() <= 1.0 else bgr,
                      0, 255).astype(np.uint8)
    h, w = bgr.shape[:2]
    tile = _memory_budget_tile(h, w)
    if tile is None:
        y = _infer_full(_SESSION["sess"], bgr)
    else:
        try:
            fixed_sess = _build_global_norm_session(path, bgr)
            y = _infer_tiled(fixed_sess, bgr, tile=tile)
        except ImportError:
            y = _infer_tiled(_SESSION["sess"], bgr, tile=tile)
    return np.clip(1.0 - y, 0.0, 1.0).astype(np.float32)
