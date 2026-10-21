# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 打包配置：单文件 lineart 可执行（Windows/Linux/macOS 通用）。
#
# 构建：pyinstaller lineart.spec --noconfirm
# 产物：dist/lineart(.exe)
#
# - 入口 run_lineart.py（顶层启动器）：包内 __main__.py 直接做入口
#   会因无父包上下文使相对导入失效，启动器经包路径调用 CLI
# - EXE 无参数（双击）自动进 tkinter GUI；带参数走 CLI
# - excludes 排除 numba 及其运行时 llvmlite（JIT 运行时自动回退纯 NumPy，
#   编译器不可随包分发；llvmlite 的 .so 约 50MB，不排除会撑爆体积）
#   与 pytest（仅测试用），控制体积与构建稳定性
# - opencv-python-headless / numpy / tkinter 由 PyInstaller 内置 hook 自动收集
# - v1.24.0：双 fp32 权重内置（datas 打包 models/：informative 17MB +
#   DexiNed 45MB 原版 blockwise DQ，无量化无降档——桌面无移动端内存
#   约束，用户指令"极致性能、两个权重完全释放"）。onefile 解包后
#   __file__ 上溯 4 级 = _MEIPASS 根，命中查找链第 3 条
#   （仓库根 models/）；~/.lineart/ 与环境变量仍可覆盖。
#   onnxruntime 加入 hiddenimports（代码内函数级动态 import，
#   静态分析兜底）确保 ORT DLL 随包分发，否则神经语义在 EXE 内不可用

a = Analysis(
    ["run_lineart.py"],
    pathex=[],
    binaries=[],
    datas=[("models", "models")],
    hiddenimports=["lineart.selfcheck", "onnxruntime"],
    hookspath=[],
    hooksconfig={},
    excludes=["numba", "llvmlite", "pytest", "setuptools", "pip"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="lineart",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
