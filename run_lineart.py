"""PyInstaller 打包入口：经包路径调用 CLI（保持相对导入有效）。

- EXE 无参数（双击）→ GUI；带参数 → CLI（见 lineart/__main__.py）
- 打包：pyinstaller lineart.spec --noconfirm
"""
import sys

from lineart.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
