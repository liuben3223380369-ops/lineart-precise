"""功能可用性自检（v1.12.0）：selfcheck 机制回归测试。

背景：v1.11.3 APK 曾出现「构建成功但运行即崩」（Chaquopy 平铺导致
lineart 包不存在），import 级冒烟测不出这类问题。selfcheck 用真实
合成小图端到端跑管线（转换/语义/场景/批量/SVG/CLI），core 项失败
即退出码 1，作为 CI 产物级门禁。
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lineart import __version__
from lineart.__main__ import _build_parser
from lineart.selfcheck import CheckResult, run_selfcheck
import lineart.selfcheck as sc


class TestSelfcheckRun(unittest.TestCase):
    """run_selfcheck 全量执行：本机全依赖环境下 core 必须全过。"""

    @classmethod
    def setUpClass(cls):
        cls.ok, cls.results = run_selfcheck()

    def test_core_all_pass(self):
        """core 11 项在本机完整依赖下必须全部 PASS（否则发布门禁红）。"""
        self.assertTrue(self.ok, msg="\n".join(
            f"{r.name}: {r.status} {r.detail}"
            for r in self.results if r.category == "core" and r.status != "PASS"))

    def test_total_14_checks(self):
        """检测项总数 14：core 11 + optional 3，防止静默增删。"""
        self.assertEqual(len(self.results), 14)
        self.assertEqual(
            sum(r.category == "core" for r in self.results), 11)
        self.assertEqual(
            sum(r.category == "optional" for r in self.results), 3)

    def test_core_item_names(self):
        """core 检测项名单固定（CI 门禁依赖这些名字出报告）。"""
        expected = {
            "deps:numpy-opencv",
            "core:convert-grayscale", "core:convert-binary",
            "core:convert-color", "core:convert-layered",
            "core:render-shape", "core:semantic-x3", "core:scene-x5",
            "core:batch-api", "core:svg-export", "core:cli-entry",
        }
        actual = {r.name for r in self.results if r.category == "core"}
        self.assertEqual(actual, expected)

    def test_checkresult_structure(self):
        """每条结果字段完整、取值合法、耗时非负。"""
        for r in self.results:
            self.assertIsInstance(r, CheckResult)
            self.assertIsInstance(r.name, str)
            self.assertIn(r.category, ("core", "optional"))
            self.assertIn(r.status, ("PASS", "FAIL", "SKIP", "INFO"))
            self.assertIsInstance(r.detail, str)
            self.assertGreaterEqual(r.ms, 0)

    def test_optional_never_fail(self):
        """optional 项缺依赖只允许 SKIP/INFO，不允许 FAIL。"""
        bad = [r.name for r in self.results
               if r.category == "optional" and r.status == "FAIL"]
        self.assertEqual(bad, [])

    def test_semantic_converters_exercised(self):
        """三语义 + 5 场景确实被端到端执行（非仅 import）。"""
        by_name = {r.name: r for r in self.results}
        self.assertIn("fdog", by_name["core:semantic-x3"].detail)
        self.assertIn("pixel", by_name["core:scene-x5"].detail)


class TestSelfcheckFailureSemantics(unittest.TestCase):
    """core 项抛异常必须被捕获并计为 FAIL → ok=False。"""

    def test_exception_becomes_fail(self):
        def _boom():
            raise RuntimeError("模拟运行期故障")

        with mock.patch.object(
                sc, "_CORE",
                [("core:fake", _boom)]):
            with mock.patch.object(sc, "_OPTIONAL", []):
                ok, results = run_selfcheck()
        self.assertFalse(ok)
        self.assertEqual(results[0].status, "FAIL")
        self.assertIn("RuntimeError", results[0].detail)
        self.assertIn("模拟运行期故障", results[0].detail)

    def test_optional_exception_not_fatal(self):
        """optional 项异常同样捕获，但不影响 core 判定。"""
        def _boom():
            raise RuntimeError("可选功能故障")

        with mock.patch.object(sc, "_CORE", []):
            with mock.patch.object(
                    sc, "_OPTIONAL",
                    [("optional:gui", _boom)]):
                ok, results = run_selfcheck()
        self.assertTrue(ok)
        self.assertEqual(results[0].status, "FAIL")


class TestCliIntegration(unittest.TestCase):
    """--selfcheck / --version 参数集成。"""

    def test_parser_has_selfcheck_flag(self):
        args = _build_parser().parse_args(["--selfcheck"])
        self.assertTrue(args.selfcheck)
        self.assertFalse(args.gui)

    def test_selfcheck_beats_gui_when_both(self):
        """--gui --selfcheck 同给时 selfcheck 生效（main 拦截在 GUI 之前）。"""
        args = _build_parser().parse_args(["--gui", "--selfcheck"])
        self.assertTrue(args.selfcheck)

    def test_version_action(self):
        """--version 打印版本并退出 0（selfcheck 的 cli-entry 依赖它）。"""
        import io
        import contextlib
        buf = io.StringIO()
        with self.assertRaises(SystemExit) as cm:
            with contextlib.redirect_stdout(buf):
                _build_parser().parse_args(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(__version__, buf.getvalue())

    def test_default_no_selfcheck(self):
        args = _build_parser().parse_args(["input.png", "-o", "out.png"])
        self.assertFalse(args.selfcheck)


if __name__ == "__main__":
    unittest.main()
