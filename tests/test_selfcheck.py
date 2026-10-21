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
        """三语义 + 5 场景确实被端到端执行（非仅 import）。

        注：合成线稿图的自动检测结果不锁定具体场景名（v1.13.0 起
        块状实心率判据使白底细线图判为 lineart，旧版本曾误判 pixel）。
        """
        by_name = {r.name: r for r in self.results}
        self.assertIn("fdog", by_name["core:semantic-x3"].detail)
        self.assertIn("自动检测=", by_name["core:scene-x5"].detail)
        self.assertIn("5 场景强制路由均成功", by_name["core:scene-x5"].detail)


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


def _mock_gui_deps():
    """构造 tkinter/PIL 假模块（MagicMock：任意属性可调用、可链式），
    使 _check_gui 能走到「import + 实例化 LineArtPreviewGUI」全程
    （CI/本地无显示环境时 tkinter 不可用会提前 SKIP）。"""
    tk = mock.MagicMock(name="tkinter")
    tk.TclError = type("TclError", (Exception,), {})  # 真实异常类便于 raise
    mods = {
        "tkinter": tk,
        "tkinter.ttk": mock.MagicMock(name="tkinter.ttk"),
        "tkinter.filedialog": mock.MagicMock(name="tkinter.filedialog"),
        "tkinter.messagebox": mock.MagicMock(name="tkinter.messagebox"),
        "PIL": mock.MagicMock(name="PIL"),
        "PIL.Image": mock.MagicMock(name="PIL.Image"),
        "PIL.ImageTk": mock.MagicMock(name="PIL.ImageTk"),
    }
    for name, m in mods.items():
        if name.startswith("tkinter."):
            setattr(tk, name.split(".", 1)[1], m)
    mods["PIL"].Image = mods["PIL.Image"]
    mods["PIL"].ImageTk = mods["PIL.ImageTk"]
    return mods


class TestCheckGuiLoadsPreviewModule(unittest.TestCase):
    """v1.24.2：_check_gui 必须真正加载 GUI 模块——此前只探测
    tkinter/PIL 可用性，preview.py 导入链错误（is_supported_video
    缺失）漏检，v1.24.0/1.24.1 EXE 双击 GUI 崩到 CI 门禁之外。
    v1.26.1：升级为实例化级——构造期错误（如 pack() 收到
    -foreground 的 TclError，v1.25.0/v1.26.0 EXE 双击即崩）也拦。"""

    def test_check_gui_pass_when_instantiation_ok(self):
        with mock.patch.dict(sys.modules, _mock_gui_deps()):
            sys.modules.pop("lineart.gui.preview", None)
            sys.modules.pop("lineart.gui", None)
            status, detail = sc._check_gui()
        self.assertEqual(status, "PASS", msg=detail)

    def test_check_gui_fail_when_import_chain_broken(self):
        # sys.modules 塞 None = 该模块导入失败（CPython 约定），
        # from .gui.preview import ... 将抛 ImportError → FAIL
        mods = _mock_gui_deps()
        mods["lineart.gui.preview"] = None
        with mock.patch.dict(sys.modules, mods):
            status, detail = sc._check_gui()
        self.assertEqual(status, "FAIL", msg=detail)
        self.assertIn("GUI 模块加载失败", detail)

    def test_check_gui_fail_when_widget_ctor_raises_tclerror(self):
        """模拟 v1.26.1 真实回归：widget 构造/pack 期抛非显示类
        TclError（如 bad option "-foreground"）→ FAIL 而非 SKIP。"""
        mods = _mock_gui_deps()
        tk = mods["tkinter"]
        tcl_err = tk.TclError
        # preview.py 经 `from tkinter import ttk` 使用 ttk.Label——
        # 即 sys.modules["tkinter"].ttk 属性（已在 _mock_gui_deps 关联）
        bad_label = mock.MagicMock()
        bad_label.pack.side_effect = tcl_err(
            'bad option "-foreground": must be -after, -anchor, ...')
        mods["tkinter.ttk"].Label = mock.MagicMock(return_value=bad_label)
        with mock.patch.dict(sys.modules, mods):
            sys.modules.pop("lineart.gui.preview", None)
            sys.modules.pop("lineart.gui", None)
            status, detail = sc._check_gui()
        self.assertEqual(status, "FAIL", msg=detail)
        self.assertIn("GUI 实例化失败", detail)
        self.assertIn("foreground", detail)

    def test_check_gui_skip_when_no_display(self):
        """无显示环境（Tk() 抛 display 类 TclError）→ SKIP 放行
        （CI Linux runner 无 X，不能误判 FAIL）。"""
        mods = _mock_gui_deps()
        tk = mods["tkinter"]
        tk.Tk = mock.MagicMock(side_effect=tk.TclError(
            "no display name and no $DISPLAY environment variable"))
        with mock.patch.dict(sys.modules, mods):
            sys.modules.pop("lineart.gui.preview", None)
            sys.modules.pop("lineart.gui", None)
            status, detail = sc._check_gui()
        self.assertEqual(status, "SKIP", msg=detail)
        self.assertIn("无显示环境", detail)


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
