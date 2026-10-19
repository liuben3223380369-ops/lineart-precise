"""Android bridge 契约测试（v1.14.0）。

背景：v1.13.0 APK 实机「转换报错」的教训——bridge 只在 CI 做过文件
存在性检查，从未被真实执行。本测试把 android/app/src/main/python/bridge.py
按路径加载，在桌面栈上锁定其行为契约（Kotlin 端 callAttr 依赖的签名与
语义、binary 纯黑白、语义先验路径、selfcheck 可用性）。

注意：APK 内运行的是 Chaquopy 锁定的旧栈（numpy 1.26.2 + opencv
4.5.1.48，py3.10），本测试用当前栈验证逻辑契约；旧栈兼容性由 CI 的
bridge 真实转换冒烟门禁 + release 产物自检覆盖。
"""
import importlib.util
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lineart.io import load_image

_BRIDGE_PATH = (Path(__file__).resolve().parents[1]
                / "android" / "app" / "src" / "main" / "python" / "bridge.py")


def _load_bridge():
    spec = importlib.util.spec_from_file_location("android_bridge", _BRIDGE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bridge = _load_bridge()


def _line_image(size=96, gray=(40, 40, 40)):
    img = np.full((size, size, 3), 255, np.uint8)
    cv2.line(img, (16, 16), (size - 16, size - 16), gray, 2)
    cv2.line(img, (size - 16, 16), (16, size - 16), gray, 2)
    return img


@pytest.fixture()
def tmp_img(tmp_path):
    p = tmp_path / "in.png"
    cv2.imwrite(str(p), _line_image())
    return str(p)


class TestConvertFile:
    def test_default_returns_size_and_ms(self, tmp_img, tmp_path):
        out = tmp_path / "out.png"
        info = bridge.convert_file(tmp_img, str(out))
        assert out.exists() and out.stat().st_size > 0
        assert info.startswith("96x96,") and info.endswith(" ms")
        res = load_image(str(out))
        assert res.shape[:2] == (96, 96)

    def test_output_size_equals_input_size(self, tmp_path):
        # API 契约：输出尺寸 = 输入尺寸（max_side=2048 仅是上限）
        p = tmp_path / "in.png"
        cv2.imwrite(str(p), _line_image(size=160))
        out = tmp_path / "out.png"
        bridge.convert_file(str(p), str(out))
        assert load_image(str(out)).shape[:2] == (160, 160)

    def test_binary_flag_yields_pure_black_white(self, tmp_img, tmp_path):
        # v1.14.0 修复：纯黑白由 mode="binary" 控制（output=binary 是灰度
        # 的历史既有行为，旧 APK 误用导致开关无效）
        out = tmp_path / "out_binary.png"
        bridge.convert_file(tmp_img, str(out), binary=True)
        res = load_image(str(out))
        assert set(np.unique(res).tolist()) <= {0, 255}

    def test_semantic_map_path_accepted(self, tmp_img, tmp_path):
        sm = tmp_path / "semantic.png"
        cv2.imwrite(str(sm), _line_image(gray=(20, 20, 20)))
        out = tmp_path / "out_sm.png"
        info = bridge.convert_file(tmp_img, str(out), semantic_map_path=str(sm))
        assert out.exists() and out.stat().st_size > 0
        assert info.startswith("96x96,")

    def test_lines_detected_not_blank(self, tmp_img, tmp_path):
        out = tmp_path / "out.png"
        bridge.convert_file(tmp_img, str(out))
        res = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert (res < 128).mean() > 0.005, "未检出任何线条"


class TestToSemanticMap:
    def test_white_bg_black_lines(self):
        # 白底黑线：背景→0（无线），线→接近 1
        sm = bridge.to_semantic_map(_line_image())
        assert sm.shape == (96, 96)
        assert sm.dtype == np.float32
        assert sm.max() > 0.5
        assert sm[:8, :8].mean() < 0.1  # 左上角是白背景

    def test_black_bg_white_lines_autoflipped(self):
        # 黑底白线：四角中值判背景后自动反相，语义方向保持一致
        img = 255 - _line_image()
        sm = bridge.to_semantic_map(img)
        assert sm.max() > 0.5
        assert sm[:8, :8].mean() < 0.1  # 反相后左上角黑背景→0

    def test_range_clipped(self):
        rng = np.random.RandomState(0)
        img = (rng.rand(64, 64, 3) * 255).astype(np.uint8)
        sm = bridge.to_semantic_map(img)
        assert sm.min() >= 0.0 and sm.max() <= 1.0


class TestSelfcheck:
    def test_selfcheck_passes(self):
        ok, report = bridge.selfcheck()
        assert ok, report
        assert "PASS" in report and "FAIL" not in report
