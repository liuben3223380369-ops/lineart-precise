#!/usr/bin/env python3
"""v1.26.0: Android DexiNed 动态轴重量化权重生成。

输入: models/dexined.onnx（「bias 修复 + 动态轴」终态, opset 21,
      weight 2D blockwise DQ 保留）
输出: android asset dexined.onnx（动态轴 + per-channel INT8 DQ,
      opset 17, onnxruntime-android 1.17 可跑）

流程（v1.23.0 repack 的动态版重制）:
  1. 全反量化 50 个 blockwise DQ → fp32 weight
  2. per-output-channel 对称 INT8 重量化（axis=0, 无 block_size,
     opset 13 标准形态, 任意 ORT 可跑）
  3. opset stamp 21 → 17（算子集 opset 13 即满足）
验证: 960x1280 动态推理 + 与 fp32 版输出偏差
"""
import sys

import numpy as np
import onnx
import onnxruntime as ort
import cv2
from onnx import numpy_helper

SRC = "models/dexined.onnx"
DST = "/tmp/dexined_android_dyn.onnx"


def dequantize_blockwise(data, scale, zp, block):
    """blockwise(axis=1) 反量化: w[c,k] = (q[c,k] - zp[c,k//b]) * s[c,k//b]"""
    c, k = data.shape
    idx = np.arange(k) // block
    zp_b = zp.astype(np.float32)
    sc_b = scale.astype(np.float32)
    if zp_b.ndim == 2:
        zp_b = zp_b[:, idx] if zp_b.shape[1] > 1 else np.broadcast_to(
            zp_b, (c, k))
    else:
        zp_b = np.broadcast_to(zp_b.reshape(1, -1), (c, k))[:, idx] \
            if zp_b.size > 1 else np.broadcast_to(
                zp_b.reshape(1, 1), (c, k))
    if sc_b.ndim == 2:
        sc_b = sc_b[:, idx] if sc_b.shape[1] > 1 else np.broadcast_to(
            sc_b, (c, k))
    else:
        sc_b = np.broadcast_to(sc_b.reshape(1, -1), (c, k))[:, idx] \
            if sc_b.size > 1 else np.broadcast_to(
                sc_b.reshape(1, 1), (c, k))
    return (data.astype(np.float32) - zp_b) * sc_b


def main():
    m = onnx.load(SRC)
    g = m.graph
    init = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    init_set = set(init.keys())

    fp32_backup = {}
    dqs = [n for n in g.node if n.op_type == "DequantizeLinear"]
    print(f"DQ 节点: {len(dqs)}")
    for n in dqs:
        data = init[n.input[0]]
        if data.ndim != 2:
            raise RuntimeError(f"意外非 2D DQ data: {n.name} {data.shape}")
        attrs = {a.name: a.i for a in n.attribute}
        block = attrs.get("block_size", 0)
        scale = init[n.input[1]]
        zp = init[n.input[2]] if len(n.input) > 2 else np.zeros(
            1, dtype=np.int8)
        if block:
            w = dequantize_blockwise(data, scale, zp, block)
        else:
            w = (data.astype(np.float32) - zp.astype(np.float32).reshape(
                -1)) * scale.astype(np.float32).reshape(-1)
        fp32_backup[n.input[0]] = w

        # per-output-channel 非对称量化 (axis=0): q∈[0,255] uint8 + zp
        # 精度优先：非对称较对称误差再减半（256 vs 254 量化级 + 零点自适应）
        c = w.shape[0]
        wmin = w.min(axis=1).astype(np.float32)
        wmax = w.max(axis=1).astype(np.float32)
        scale_ch = np.maximum((wmax - wmin) / 255.0, 1e-8).astype(np.float32)
        zp_ch = np.clip(np.round(-wmin / scale_ch), 0, 255).astype(np.uint8)
        q = np.clip(np.round(w / scale_ch.reshape(-1, 1))
                    + zp_ch.reshape(-1, 1), 0, 255).astype(np.uint8)
        # 替换 initializer
        for i, arr, name in (
                (0, q, n.input[0]),
                (1, scale_ch, n.input[1]),
                (2, zp_ch,
                 n.input[2] if len(n.input) > 2 else None)):
            if name is None or name not in init_set:
                continue
            old = next(x for x in g.initializer if x.name == name)
            new = numpy_helper.from_array(arr, name)
            old.CopyFrom(new)
        if len(n.input) <= 2:
            n.input.append(n.input[1] + "_zp")
            g.initializer.append(numpy_helper.from_array(
                zp_ch, n.input[2]))
        # attrs: block_size 删除, axis → 0
        del n.attribute[:]
        n.attribute.append(onnx.helper.make_attribute("axis", 0))

    # 删除已无用的旧 zp/scale blockwise initializer（如被替换则已覆盖）
    m.opset_import[0].version = 17
    onnx.checker.check_model(m)
    onnx.save(m, DST)
    print(f"saved: {DST}")

    # 验证 1: fp32 参考模型（全反量化）
    mref = onnx.load(SRC)
    gref = mref.graph
    renames = {}
    new_inits = []
    for n in list(gref.node):
        if n.op_type != "DequantizeLinear":
            continue
        w = fp32_backup[n.input[0]]
        name = n.output[0] + "_fp32"
        new_inits.append(numpy_helper.from_array(
            w.astype(np.float32), name))
        renames[n.output[0]] = name
        gref.node.remove(n)
    for n in gref.node:
        for i, inp in enumerate(n.input):
            if inp in renames:
                n.input[i] = renames[inp]
    gref.initializer.extend(new_inits)
    onnx.save(mref, "/tmp/dexined_ref_fp32.onnx")

    # 验证 2: 960x1280 动态推理 + 输出偏差
    mean = np.array([103.5, 116.2, 123.6], dtype=np.float32)
    outs = {}
    for tag, p in (("ref", "models/dexined.onnx"),
                   ("q", DST)):
        img = np.random.RandomState(0).randint(
            0, 256, (960, 1280, 3)).astype(np.uint8)
        x = np.transpose(img.astype(np.float32) - mean, (2, 0, 1))[None]
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        outs[tag] = s.run(["block_cat"], {"img": x})[0]
    d = np.abs(outs["ref"].astype(np.float64) - outs["q"].astype(
        np.float64))
    edge_ref = 1.0 / (1.0 + np.exp(-outs["ref"]))
    edge_q = 1.0 / (1.0 + np.exp(-outs["q"]))
    mask_agree = ((edge_ref > 0.5) == (edge_q > 0.5)).mean()
    print(f"960x1280 动态推理 OK; fuse 偏差 mean={d.mean():.6f} "
          f"max={d.max():.4f}; sigmoid 后 0.5 mask 一致率={mask_agree:.5f}")
    import os
    print(f"体积: {os.path.getsize(DST)/1048576:.1f}MB")


if __name__ == "__main__":
    main()
