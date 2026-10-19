# lineart-precise v1.13.0

高精度「图片 → 白底黑线」线稿转换器：亚像素级线条定位、场景自适应、外形保真双渲染路线。本版本起提供免安装桌面版与 Android 版。

## 下载

| 文件 | 平台 | 说明 |
|---|---|---|
| `lineart-1.13.0-windows-x64.exe` | Windows 10/11 x64 | 免安装单文件；双击进 GUI，命令行当 CLI |
| `lineart-1.13.0-android-arm64.apk` | Android 8.0+（arm64） | 选图 → 调参 → 转换 → 保存，管线与桌面版同源 |
| `lineart-1.13.0-linux-x64` | Linux x86-64（glibc 2.28+） | 免安装单文件 CLI |

## 系统要求

- **Windows**：Win10+ x64，无需安装 Python 或任何依赖（单文件含 Python + OpenCV + NumPy 全套运行时，v1.11.6 实测：Windows 64.5MB / Linux 99.3MB）
- **Android**：8.0+（API 24）、arm64-v8a 架构，APK 约 35MB（内含 Python 运行时与 OpenCV）
- **Linux**：glibc 2.28+（Ubuntu 20.04+ / CentOS 8+ / Debian 11+）

## 安装步骤

### Windows

1. 下载 `lineart-1.13.0-windows-x64.exe`，放到任意目录
2. **图形界面**：直接双击，进入 12 参数滑动条的实时预览 GUI
3. **命令行**：
   ```bat
   lineart.exe photo.jpg -o lineart.png
   lineart.exe ./photos -o ./results --batch --workers 4
   ```
4. 首次运行若弹出 SmartScreen 譝告（未购买代码签名证书的正常提示）：点「更多信息」→「仍要运行」

### Android

1. 下载 `lineart-1.13.0-android-arm64.apk`，用文件管理器打开
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
## v1.13.0（2026-10-19）：转换质量与 GUI 大图查看修复

**背景**：用户实测反馈三类问题——黑底白线 CAD 图转换失真严重（双线/文字重影）、生活照片转换轮廓大量缺失、GUI 无法查看大图。全部复现定位并修复。

### 修复

- **黑底白线 CAD 误判 pixel 场景**（失真主因）：此类图「颜色少 + 硬边」与像素艺术在统计上无法区分，被误判后套用 4x 预放大 + thickness=4 + 不合并边缘，产生双线化、文字重影、实心点空心化，笔画保留率仅 5%。新增**块状实心率判据**（`_blocky_solid_ratio`，距离变换：像素艺术大色块内部距背景 ≥2px，实心率 0.97；1~3px 细线笔画 ≤0.18，阈值 0.5）——CAD 黑底图现正确路由 lineart 场景，笔画保留率恢复 111%，文字清晰可辨
- **场景默认参数静默覆盖用户调参**（照片调优障碍）：`_apply_scene` 的场景值无条件覆盖用户参数（photo 场景 denoise=0.55 顶掉用户手动值，照片弱边缘完全无法调优）。改为**场景值只覆盖仍等于默认值的字段**，用户改过的参数优先
- **GUI 大图查看**：缩放上限 5x → 32x（旧上限下 4K 图最大只能放到 ~1.3 倍原始尺寸，无法检查线条质量）；新增左键拖拽平移（scrollregion）；放大切换 INTER_NEAREST 保持像素锐利（AREA 放大模糊）；新增「适应窗口」「1:1 原始像素」按钮
- **`normalize_illumination` uint8 输入防御**：strength<1 分支对 [0,1] 结果与 [0,255] 输入直接混合会把整图推成全白；入口统一归一化到 [0,1] float

### 照片转换说明（已知边界）

照片线稿的核心矛盾是「纹理抑制 vs 轮廓保留」。实测直接放宽 photo 场景参数（denoise/gamma）轮廓恢复有限（<5%）而地板纹理噪声增长 8.8 倍，故本版本不改 photo 场景默认值，而是**解锁用户调参能力**（上面的覆盖修复）：照片弱边缘建议 `--denoise 0.2 --sensitivity 0.7` 起步微调。渲染层的拐角断裂/节点问题列入后续版本。

### 测试

新增 `tests/test_scene_routing.py` 14 项（块状实心率区分度、黑底 CAD 路由、官方样本零回归、参数覆盖语义、uint8 防御）；全量 64 项 + 20 子测试、perf_v13 逐位一致、60 项压测、selfcheck 14 项全过。

## v1.12.0（2026-10-19）：功能可用性自检

**背景**：v1.11.3 APK 曾出现「构建成功但运行即崩」（Chaquopy srcDir 平铺导致 lineart 包不存在），import 级冒烟测不出这类问题。本版本建立端到端功能门禁：**跑真实小样例，而非仅 import**。

### 新增

- **`--selfcheck` 功能可用性自检**（`lineart/selfcheck.py`，255 行）：14 项端到端检测，内置合成测试图（十字线 + 圆），每项独立 try/except，核心项失败退出码 1
  - 核心 11 项：依赖版本、灰度/纯黑白/彩色/分层四种转换、shape 渲染路线、三语义（fdog/xdog/adaptive）、5 场景强制路由（pixel/blueprint/sketch/lineart/photo）、批量 API、SVG 导出、CLI 入口
  - 可选 3 项（缺依赖仅 SKIP 不算失败）：GUI（tkinter+PIL）、视频编解码（MJPG）、numba JIT
- **`--version` 参数**：打印版本号退出（此前缺失，自检的 CLI 入口检测依赖它）
- **CI 产物级门禁**：Windows/Linux 构建后用 `dist/lineart(.exe) --selfcheck` 直接跑打包产物（Android 已有 app.imy 解包校验），杜绝「构建成功 ≠ 运行可用」
- **12 项 selfcheck 单测**（`tests/test_selfcheck.py`）：全量执行、检测项名单锁定、异常捕获语义（core 抛异常→FAIL、optional 异常不致命）、CLI 参数集成

### 使用

```bash
lineart --selfcheck          # 桌面版 / python -m lineart --selfcheck
lineart.exe --selfcheck      # Windows EXE
```

### 修复（自检首轮运行即抓出的 API 假设错误）

- binary 检测改用 `mode="binary"`（纯黑白由 mode 控制，`output="binary"` 走灰度分支为既有行为）
- layered 检测适配真实结构（mask 为 bool、color 为 BGR 元组）
- 场景检测用 `SceneInfo.scene`（非 `.name`）、批量 API 生成器语义（需 `list()` 包装）

### v1.11.6
- **项目结构优化**（仓库卫生，零功能变更）：`perf_v13.py` / `stress_test.py` 归位 `benchmarks/`（压测为模块级自执行，放 tests/ 会被 pytest 误收集执行）；清除 7 处 `sys.path.insert` 与 2 处样图绝对路径硬编码，全部改为基于 `__file__` 的相对定位（任意机器/任意 cwd 可跑）；`demo/output/` 83 个生成物移出版本库；`.gitignore` 补全（.pytest_cache/build/dist/android 构建产物等）
- 验证：38 项单测 + 15 子测试、perf_v13 六节精度逐位一致、60 项压测全过、CLI/打包冒烟正常



### v1.11.5
- **修正 CI 校验方式**：Chaquopy 15 将应用 Python 源码打包为 `assets/chaquopy/app.imy` 归档（非散文件），v1.11.4 的校验在 APK 散文件里找 `lineart/__init__.py` 导致误判失败（修复本身已生效，构建产物正常）；现改为解包 `app.imy` 检查内部条目（`lineart/__init__.py`、`lineart/pipeline.py`、`bridge.py`）



### v1.11.4
- **修复 Android 启动崩溃**（`No module named 'lineart'`）：Chaquopy `srcDir` 语义是「目录内容平铺到 Python 顶层」，此前直接指向 `lineart` 包目录导致包内模块被平铺、APK 内无 `lineart` 包（构建能过、运行必崩）；现改为 Gradle `Sync` task 把包完整复制到 `build/lineart-pkg/lineart/` 后作为源码目录，包结构完整进 APK
- **CI 增强**：Android 产物新增「APK 内含 lineart 包」校验（unzip 检查 `lineart/__init__.py`），杜绝此类装配错误静默过检



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
