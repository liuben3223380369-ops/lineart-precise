# lineart-precise v1.11.3

高精度「图片 → 白底黑线」线稿转换器：亚像素级线条定位、场景自适应、外形保真双渲染路线。本版本起提供免安装桌面版与 Android 版。

## 下载

| 文件 | 平台 | 说明 |
|---|---|---|
| `lineart-1.11.3-windows-x64.exe` | Windows 10/11 x64 | 免安装单文件；双击进 GUI，命令行当 CLI |
| `lineart-1.11.3-android-arm64.apk` | Android 8.0+（arm64） | 选图 → 调参 → 转换 → 保存，管线与桌面版同源 |
| `lineart-1.11.3-linux-x64` | Linux x86-64（glibc 2.28+） | 免安装单文件 CLI |

## 系统要求

- **Windows**：Win10+ x64，无需安装 Python 或任何依赖（单文件含 Python + OpenCV + NumPy 全套运行时，v1.11.3 实测：Windows 64.5MB / Linux 99.3MB）
- **Android**：8.0+（API 24）、arm64-v8a 架构，APK 约 35MB（内含 Python 运行时与 OpenCV）
- **Linux**：glibc 2.28+（Ubuntu 20.04+ / CentOS 8+ / Debian 11+）

## 安装步骤

### Windows

1. 下载 `lineart-1.11.3-windows-x64.exe`，放到任意目录
2. **图形界面**：直接双击，进入 12 参数滑动条的实时预览 GUI
3. **命令行**：
   ```bat
   lineart.exe photo.jpg -o lineart.png
   lineart.exe ./photos -o ./results --batch --workers 4
   ```
4. 首次运行若弹出 SmartScreen 譝告（未购买代码签名证书的正常提示）：点「更多信息」→「仍要运行」

### Android

1. 下载 `lineart-1.11.3-android-arm64.apk`，用文件管理器打开
2. 首次安装按系统提示允许「安装未知应用」权限
3. 打开「LineArt 线稿」→ 选择图片 → 调节细节/灵敏度/粗细 → 转换 → 保存 PNG
4. 界面还提供「纯黑白输出」「外形保真渲染」两个开关（对应桌面版 `--mode binary` 与 `--render-mode shape`）

## 快速上手（CLI）

```bash
# 单张（默认 FDoG 流线语义）
lineart.exe photo.jpg -o lineart.png

# 纯黑白 + 加粗
lineart.exe photo.jpg -o out.png --mode binary --thickness 2

# 外形保真路线（宽线/变宽线/笔触保真，制版推荐）
lineart.exe photo.jpg -o out.png --render-mode shape --shape-ink source

# 目录批量（多线程）
lineart.exe ./photos -o ./results --batch --workers 4

# SVG 矢量输出
lineart.exe photo.jpg -o out.svg --output-svg
```

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | antialiased | `antialiased` 抗锯齿灰度 / `binary` 纯黑白 |
| `--semantic` | fdog | `fdog` / `xdog` / `adaptive` |
| `--scene` | auto | `auto`/`off`/`pixel`/`blueprint`/`sketch`/`lineart`/`photo` |
| `--render-mode` | splat | `splat` 中心线重绘 / `shape` 外形保真 |
| `--detail` | 0.6 | 细节保持 |
| `--sensitivity` | 0.5 | 线条灵敏度 |
| `--thickness` | 1.0 | 线条粗细（倍率） |
| `--coherence` | 0.6 | 流线一致性 |
| `--max-side` | 4096 | 最长边限制 |
| `--batch` | - | 目录批量 + `--workers N` |

## 已知问题

- Windows SmartScreen 警告：未购买代码签名，属正常提示，选「仍要运行」
- APK 使用 CI 演示签名：适合个人使用；若需上架商店请自行重签
- numba JIT 加速未编入安装包：自动回退纯 NumPy，功能不受影响
- Android 端超大图（>4096px）耗时较长，已在后台线程执行，请耐心等待状态栏提示

## 版本日志

### v1.11.3
- **修复 Android 编译**：Kotlin 调用 Chaquopy 的 API 名为 `callAttr`（Java 风峰）而非 `call_attr`，`MainActivity.kt` 已更正

### v1.11.2
- **修复 Android 构建**：numpy/opencv 版本此前不在 Chaquopy 预编译索引（`chaquo.com/pypi-13.1`）内，pip 退回源码构建失败；实测索引修正为 `numpy==1.26.2` + `opencv-python==4.5.1.48`（均为 cp310/android arm64-v8a wheel，lineart 所用 cv2 API 全部兼容 4.5.1）

### v1.11.1
- **修复 Windows CLI 崩溃**：EXE 在 cp1252 控制台打印中文结果时 UnicodeEncodeError 退出（转换本身成功）；启动器现强制 stdout/stderr 为 UTF-8
- **修复 Android CI 失败**：移除多余 `setup-android` step（ubuntu-latest 预装 Android SDK，其 sdkmanager 在 runner 上异常）

### v1.11.0
- **一键打包发布**：GitHub Actions 云端构建 Windows EXE / Linux 单文件 / Android APK（tag `v*` 触发，本页三附件即产物）
- **EXE 双击即用**：无参数启动自动进入实时预览 GUI，带参数仍走 CLI
- **Android 版**：Kotlin UI + Chaquopy 编入同源 Python 管线（OpenCV/NumPy 预编译），细节/灵敏度/粗细实时调参，支持 binary 与 shape 渲染
- PyInstaller 单文件 Linux 实测约 80MB（cv2 16.6M + libpython 11.2M + OpenBLAS 13M + 视频编解码 14.5M；excludes numba/llvmlite/pytest 等，CLI 冒烟 988ms）

### v1.10.0
- shape 外形保真渲染路线：宽线 IoU 0.199→1.000、变宽线 corr 0.619→1.000、圆径向误差 0.065→0.012px、4096² 提速 2 倍

### v1.9.2
- 桥接 O(N²)→空间网格（65s→0.36s）、Bresenham 斜线 bug 修复、移除无效 ETF 调用

### v1.10.1
- 项目结构文档（codebase-structure.md）、README 史实修正（场景 5 种、测试 38 项、压测 60 项）
