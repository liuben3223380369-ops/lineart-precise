"""PyInstaller 打包入口：经包路径调用 CLI（保持相对导入有效）。

- EXE 无参数（双击）→ GUI；带参数 → CLI（见 lineart/__main__.py）
- 打包：pyinstaller lineart.spec --noconfirm
- Windows 控制台默认 cp1252，中文输出会 UnicodeEncodeError，
  这里强制 stdout/stderr 为 UTF-8（errors=replace 兜底）
"""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from lineart.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
