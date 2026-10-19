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

a = Analysis(
    ["run_lineart.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=["lineart.selfcheck"],
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
