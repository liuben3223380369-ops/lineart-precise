"""DexiNed INT8 边缘语义（semantic="dexined"，v1.20.0 内置）。

OpenCV 官方 ONNX 动物园量化版（opencv/edge_detection_dexined，
45MB INT8 blockwise，BIPED SOTA 训练）。实测精度优于
informative-drawings：合成圆原始输出 std 0.3756→0.0837，
端到端定位偏差 -0.075→-0.042，弱对比山脊召回 80.6%→91.9%。

两层 ONNX 图手术（均一次性、缓存复用）：
1. bias 反量化（.fixed.onnx）：原始权重 bias 的 DequantizeLinear 带
   block_size 属性，ORT 1.31 抛 "block_size must be 0" 异常。把 1D
   bias DQ 反量化为 fp32 initializer（数值等价），weight 的 2D
   blockwise DQ 保留（INT8 精度无损，与全 fp32 反量化版输出逐位一致）。
2. 动态尺寸（.dyn.onnx，v1.21.0）：网络为纯全卷积（Reshape 硬编码
   均为权重形状、ConvTranspose 无 output_shape 硬编码），将输入/输出
   的 480/640 dim 改为动态参数后任意分辨率可直推。scale=2 推理
   分辨率 = 2x 原图并夹在 [480×640, 960×1280]（小图防低于训练
   分辨率精度下降、大图防 OOM，960×1280 峰值内存 ~2.7GB）。
   大图清晰硬边缘收益显著（定位偏差 -43%、锐度 +95%），弱对比
   软边缘召回下降（-12pp，感受野失配）→ 场景路由由管线层
   （dexined_scale="auto"：photo/sketch→1x，lineart/blueprint/pixel→2x）。

v1.22.0 诊断确认本模型响应分布健康（阈值 0.3→0.6 召回仅降 0.10），
是 ensemble 融合（ensemble.py）的精度基准分量。
"""
import os
from pathlib import Path

import cv2
import numpy as np

from .informative import _try_ort

_DEXINED_NAME = "dexined.onnx"
_DEXINED_SOURCE = ("https://hf-mirror.com/opencv/"
                   "edge_detection_dexined (resolve/main/"
                   "edge_detection_dexined_2024sep.onnx)")
_DEXINED_H, _DEXINED_W = 480, 640  # 训练分辨率（实测低于此分辨率精度下降）
_DEXINED_MEAN = np.array([103.5, 116.2, 123.6], dtype=np.float32)  # BGR

_DEXINED_SESSION = {"path": None, "sess": None}
_DEXINED_DYN_SESSION = {"path": None, "sess": None}


def _round16(v):
    """向上取整到 16 的倍数（DexiNed 动态版输入约束，v1.27.2）。"""
    return (v + 15) // 16 * 16


def _dexined_candidate_paths(model_path=None):
    if model_path:
        return [Path(model_path)]
    paths = []
    env = os.environ.get("LINEART_DEXINED_MODEL")
    if env:
        paths.append(Path(env))
    paths.append(Path.home() / ".lineart" / _DEXINED_NAME)
    paths.append(Path(__file__).resolve().parent.parent.parent.parent
                 / "models" / _DEXINED_NAME)
    return paths


def _find_dexined_model(model_path=None):
    for p in _dexined_candidate_paths(model_path):
        if p.is_file():
            return str(p)
    return None


def dexined_available():
    """DexiNed 边缘语义是否可用（onnxruntime 已装 + 权重可寻）。"""
    return _try_ort() and _find_dexined_model() is not None


def _fix_dexined_bias_quant(model_path):
    """修复 DexiNed INT8 权重的 onnxruntime 1.31 兼容性（一次性图手术）。

    OpenCV 官方量化版的 bias DequantizeLinear 带 block_size 属性，
    ORT 1.31 对非整块 bias 抛 "block_size must be 0" 异常。修复：
    把 1D bias 的 DQ 节点反量化为 fp32 initializer（数值等价），
    weight 的 2D blockwise DQ 保留（ORT 支持且 INT8 精度无损，
    实测与全 fp32 反量化版输出逐位一致）。

    返回修复后模型路径（同目录 .fixed.onnx 缓存，已存在直接复用）。
    需要 onnx 包；不可用时返回原路径（推理会在 ORT 抛错处失败）。
    """
    fixed = Path(model_path).with_suffix(".fixed.onnx")
    if fixed.is_file():
        return str(fixed)
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError:
        return model_path
    m = onnx.load(model_path)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    renames = {}
    new_inits = []
    for n in list(g.node):
        if n.op_type != "DequantizeLinear":
            continue
        data = init.get(n.input[0])
        if data is None or data.ndim != 1:  # 只处理 bias（1D）
            continue
        attrs = {a.name: a.i for a in n.attribute}
        block = attrs.get("block_size", 0)
        scale = init[n.input[1]]
        zp = (init[n.input[2]] if len(n.input) > 2
              else np.zeros(1, dtype=np.int8))
        d = data.astype(np.float32)
        if scale.size == 1:
            fp32 = (d - float(zp.flatten()[0])) * float(scale.flatten()[0])
        elif scale.size == d.shape[0]:
            fp32 = (d - zp.astype(np.float32)) * scale.astype(np.float32)
        else:  # blockwise
            idx = np.arange(d.shape[0]) // block
            fp32 = (d - zp[idx].astype(np.float32)) * scale[idx].astype(np.float32)
        name = n.output[0] + "_fp32"
        new_inits.append(numpy_helper.from_array(fp32.astype(np.float32), name))
        renames[n.output[0]] = name
        g.node.remove(n)
    if not renames:
        return model_path  # 无需修复（已是修复版或未量化）
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in renames:
                n.input[i] = renames[inp]
    used = set()
    for n in g.node:
        used.update(n.input)
    old = set(init.keys())
    keep = [i for i in g.initializer if i.name not in old or i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    g.initializer.extend(new_inits)
    try:
        onnx.save(m, str(fixed))
        return str(fixed)
    except OSError:
        return model_path  # 只读目录等，回退原路径


def _dexined_dynamic_model(path):
    """图手术将 DexiNed 固定 480x640 输入/输出改为动态尺寸（v1.21.0）。

    网络为纯全卷积：Reshape 硬编码均为权重形状（与空间无关）、
    ConvTranspose 用 pads/output_padding 推导输出尺寸。将输入与
    全部输出的 dim_value in (480, 640) 替换为 dim_param 即可任意
    分辨率直推（256x320 ~ 1280x960 实测通过）。

    返回动态版模型路径（同目录 .dyn.onnx 缓存，已存在直接复用）；
    onnx 包不可用或手术失败返回 None（调用方回退 scale=1）。

    v1.24.0：仓库内置 models/dexined.onnx 已是「bias 修复 + 动态轴」
    终态（EXE 自包含开箱即跑，无需 onnx 包图手术）。此时输入无固定
    480/640 dim、手术无改动——识别为已动态形态直接返回原路径，
    而非误判"无需手术"回退 scale=1。
    """
    dyn = Path(path).with_suffix(".dyn.onnx")
    if dyn.is_file():
        return str(dyn)
    try:
        import onnx
    except ImportError:
        return None
    try:
        m = onnx.load(path)

        def _dyn(tensor, tag):
            dims = tensor.type.tensor_type.shape.dim
            changed = False
            for i, d in enumerate(dims):
                if d.dim_value in (_DEXINED_H, _DEXINED_W):
                    dims[i].dim_param = f"{tag}_{i}"
                    changed = True
            return changed

        changed = _dyn(m.graph.input[0], "in")
        for o in m.graph.output:
            changed = _dyn(o, "out") or changed
        if not changed:
            in_dims = m.graph.input[0].type.tensor_type.shape.dim
            if len(in_dims) >= 4 and (in_dims[2].dim_param
                                      or in_dims[3].dim_param):
                return path  # 已是动态形态（v1.24.0 内置终态），直接用
            return None
        onnx.save(m, str(dyn))
        return str(dyn)
    except Exception:
        return None


def dexined_semantic(img_bgr, model_path=None, scale=1):
    """DexiNed INT8 边缘检测 ONNX 推理 → ink 语义图（[0,1]，1=线）。

    OpenCV 官方量化模型（45MB INT8，BIPED 训练）。BGR 均值减法；
    fuse 输出过 sigmoid，1=边，与管线 ink 约定一致（无需反相）。
    输出 resize 回原尺寸。

    scale（v1.21.0）：
    - 1（默认）：输入 resize 到训练分辨率 480x640（低于此分辨率
      精度下降：192 直推 std 0.155 vs resize 0.084）——向后兼容。
    - 2：动态图手术版高分辨率推理，推理分辨率 = 2x 原图并夹在
      [480x640, 960x1280] 区间（小图防低于训练分辨率、大图防 OOM，
      960x1280 峰值内存 ~2.7GB）。清晰硬边缘定位/锐度显著提升
      （大圆定位偏差 -43%、锐度 +95%），弱对比软边缘召回下降
      （-12pp）→ 场景路由由管线层决定（dexined_scale="auto"）。
      需 onnx 包做图手术；不可用时自动回退 scale=1（v1.24.0 起仓库
      models/ 内置权重已预手术为动态终态，无需 onnx 包即支持 scale=2）。

    与 informative-drawings（semantic="neural"）对比（v1.20.0 实测）：
    - 合成圆原始输出 std 0.3756→0.0837（4.5x 更稳）
    - 端到端 mean_err -0.075→-0.042（改善 44%）
    - 弱对比山脊召回 80.6%→91.9%（+11.3pp）
    - CPU 480x640 推理 2.7s vs 2.0s（1.37x 慢，秒级可接受；
      scale=2 约 3.6x 于 scale=1）

    ORT 1.31 的 bias blockwise 量化 bug：1D bias DQ 反量化为 fp32
    initializer（缓存 .fixed.onnx，需 onnx 包）。v1.24.0 起仓库内置
    models/dexined.onnx 已是「bias 修复 + 动态轴」终态——EXE/源码
    均开箱即跑 ORT 1.31，无需 onnx 包；仅用户自带原版权重时才在
    首次加载触发图手术。Session 按权重路径+scale 缓存。
    """
    import onnxruntime as ort
    if scale not in (1, 2):
        raise ValueError(f"scale 仅支持 1/2，收到 {scale!r}")
    path = _find_dexined_model(model_path)
    if path is None:
        raise FileNotFoundError(
            "未找到 DexiNed 边缘检测模型权重（semantic=\"dexined\"）。请将 "
            f"OpenCV 官方量化 ONNX（约 45MB，文件名 {_DEXINED_NAME}）"
            "放置于以下任一位置：① 环境变量 LINEART_DEXINED_MODEL 指定的路径；"
            "② ~/.lineart/dexined.onnx；③ 仓库根目录 models/dexined.onnx；"
            "或通过 params.neural_model / CLI --neural-model 显式指定。"
            f"权重来源：{_DEXINED_SOURCE}")
    path = _fix_dexined_bias_quant(path)
    bgr = np.asarray(img_bgr)
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr * 255.0 if bgr.max() <= 1.0 else bgr,
                      0, 255).astype(np.uint8)
    h, w = bgr.shape[:2]
    if scale == 2:
        dyn = _dexined_dynamic_model(path)
        if dyn is not None:
            if _DEXINED_DYN_SESSION["path"] != dyn:
                _DEXINED_DYN_SESSION["sess"] = ort.InferenceSession(
                    dyn, providers=["CPUExecutionProvider"])
                _DEXINED_DYN_SESSION["path"] = dyn
            # v1.27.2：推理尺寸 16 倍数对齐——DexiNed 多 block 上下
            # 采样路径倍数不同（4/2 混合），非 16 倍数输入内部 Concat
            # 维度不匹配直接崩溃（实测 605→"608 vs 606"，610/612/616/
            # 620 全崩；608/624/480×640/960×1280 全过）。夹取值 640/
            # 1280/480/960 均为 16 倍数，round16 后仍在区间内
            inp_w = _round16(min(max(w * 2, _DEXINED_W), _DEXINED_W * 2))
            inp_h = _round16(min(max(h * 2, _DEXINED_H), _DEXINED_H * 2))
            inp = cv2.resize(bgr, (inp_w, inp_h)).astype(np.float32)
            inp -= _DEXINED_MEAN
            x = np.transpose(inp, (2, 0, 1))[None]
            fuse = _DEXINED_DYN_SESSION["sess"].run(
                ["block_cat"], {"img": x})[0][0, 0]
            edge = 1.0 / (1.0 + np.exp(-fuse))
            if (h, w) != (inp_h, inp_w):
                edge = cv2.resize(edge, (w, h))
            return np.clip(edge, 0.0, 1.0).astype(np.float32)
        # onnx 包不可用：图手术失败，回退 scale=1
    if _DEXINED_SESSION["path"] != path:
        _DEXINED_SESSION["sess"] = ort.InferenceSession(
            path, providers=["CPUExecutionProvider"])
        _DEXINED_SESSION["path"] = path
    inp = cv2.resize(bgr, (_DEXINED_W, _DEXINED_H)).astype(np.float32)
    inp -= _DEXINED_MEAN
    x = np.transpose(inp, (2, 0, 1))[None]
    fuse = _DEXINED_SESSION["sess"].run(["block_cat"], {"img": x})[0][0, 0]
    edge = 1.0 / (1.0 + np.exp(-fuse))
    if (h, w) != (_DEXINED_H, _DEXINED_W):
        edge = cv2.resize(edge, (w, h))
    return np.clip(edge, 0.0, 1.0).astype(np.float32)
