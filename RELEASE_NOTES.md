# lineart-precise v1.26.1

高精度「图片 → 白底黑线」线稿转换器：亚像素级线条定位、场景自适应、外形保真双渲染路线、三级神经语义（neural / dexined / ensemble 三语义 noisy-OR 融合）。免安装桌面版与 Android 版随 Releases 发布；**v1.24.0 起桌面 EXE 内置双 fp32 神经权重与 onnxruntime 运行时，`--semantic ensemble` 开箱即用**。

## 下载

| 文件 | 平台 | 说明 |
|---|---|---|
| `lineart-1.26.1-windows-x64.exe` | Windows 10/11 x64 | 免安装单文件；双击进 GUI，命令行当 CLI；内置双神经权重 + ORT |
| `lineart-1.26.1-android-arm64.apk` | Android 8.0+（arm64） | 选图 → 调参 → 转换 → 保存，管线与桌面版同源；双权重 ensemble 默认 |
| `lineart-1.26.1-linux-x64` | Linux x86-64（glibc 2.28+） | 免安装单文件 CLI；内置双神经权重 + ORT |

## 系统要求

- **Windows**：Win10+ x64，无需安装 Python 或任何依赖（单文件含 Python + OpenCV + NumPy + onnxruntime 全套运行时与双 fp32 神经权重；v1.24.0 起体积明显增大——神经能力完全释放的代价，下载一次后续零依赖）
- **Android**：8.0+（API 24）、arm64-v8a 架构，APK 约 70MB（内含 Python 运行时、OpenCV 与双权重）
- **Linux**：glibc 2.28+（Ubuntu 20.04+ / CentOS 8+ / Debian 11+）

## 安装步骤

### Windows

1. 下载 `lineart-1.26.1-windows-x64.exe`，放到任意目录
2. **图形界面**：直接双击，进入 12 参数滑动条的实时预览 GUI
3. **命令行**：
   ```bat
   lineart.exe photo.jpg -o lineart.png
   lineart.exe ./photos -o ./results --batch --workers 4
   ```
4. 首次运行若弹出 SmartScreen 譝告（未购买代码签名证书的正常提示）：点「更多信息」→「仍要运行」

### Android

1. 下载 `lineart-1.26.1-android-arm64.apk`，用文件管理器打开
2. 首次安装按系统提示允许「安装未知应用」权限
3. 打开「LineArt 线稿」→ 选择图片 → 调节细节/灵敏度/粗细 → 转换 → 保存 PNG
4. 界面提供「纯黑白输出」「外形保真渲染」开关、「神经先验（可选增强）」卡片与「运行诊断自检」入口；手机相册 HEIC 照片与 EXIF 旋转已原生支持

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
## v1.26.1（2026-10-21）：Windows EXE 双击 GUI 崩溃修复——pack 参数错位

**根因**（用户 v1.25.0 EXE 双击实测）：`gui/preview.py` 视图控制栏把 `foreground="#666"` 传给了 `.pack()`——pack 只接受布局选项（-side/-fill/-padx…），收到 widget 选项即抛 `TclError: bad option "-foreground"`。**与外部环境无关**：单文件 EXE 自带 Python/tkinter/PIL/OpenCV 全套运行时，这是纯代码参数错位，任何机器上必崩。

- 此 bug 自 GUI 视图控制栏引入（v1.13.0）即存在，但长期被更早的 import 链错误（v1.24.2 才修复的 `is_supported_video`）挡住——修好前一个才暴露后一个
- 修复：`foreground` 移回 `ttk.Label(...)` 构造参数
- **门禁再强化**（v1.24.2 的教训延伸）：selfcheck `optional:gui` 从「import 模块」升级为「真实实例化 GUI」（Tk root + 构造 `LineArtPreviewGUI` + 销毁）——import 不触发 `_create_widgets`，构造期错误（pack 参数、widget 选项错位）只有实例化才能暴露。无显示环境（CI Linux）SKIP 放行；有桌面环境（CI Windows、用户机器）真实构造
- 契约测试 +2：widget 构造期抛非显示类 TclError → FAIL；`Tk()` 抛 display 类错误 → SKIP（防 CI Linux 误拦）
- Android 无改动（versionCode 23 仅为覆盖安装便利顺带 bump）

## v1.26.0（2026-10-21）：Android 精度架构对齐桌面端——放大模糊与噪点根治

**背景**（用户 v1.25.0 实测反馈）：「转换后的图片放大后细节模糊，噪点严重」。代码细查 + 与桌面 Python 端逐项比对，定位 **4 个结构性精度缺陷**（Android 神经链自 v1.23.0 起与桌面端存在代差）：

### 1. 输入链路：跳级采样 + 2048 上限（模糊+噪点双重根因）

- `decodeBitmap` 的 `setTargetSampleSize` 是整数跳级采样：2500px 图被砍半到 1250px（不是 2048！）；API<28 路径 2 的幂跳级同样粗暴。**用户放大看细节时输出原生只有 1~2K，必糊**；跳级采样还引入混叠伪纹（梯度/模型双重响应→噪点）
- 修复：`setTargetSize` 精确缩放（ImageDecoder 高质量滤波）；API<28 粗采样后双线性精修。输入上限 2048→**4096**（与桌面 CLI `--max-side` 默认一致；用户 5GB 内存不在意耗时）

### 2. informative 分块推理：384 小块 InstanceNorm 块内统计偏差（块状噪点根因）

- 桌面端 v1.20.0 已用「全局归一化图手术」（DS=896 统计 → InstanceNorm 固定仿射 → 分块，IoU 0.80→0.89）；Android 端从未对齐——384 小块各自做 InstanceNorm，**块间响应分布不一致** → 块状伪影 + 背景误报
- Kotlin 侧无法做运行时图手术（需 onnx 包）；**块 384→896** 是最优务实逼近（块内统计逼近全局：Python 实测 tile=384 IoU 0.7986 / 640 0.8810 / 整图 0.94+）。896 块 fp32 激活峰值 ~1.2GB（5GB 预算内）

### 3. DexiNed 固定 480×640 推理（大图细节糊带根因）

- 桌面端 v1.21.0 场景路由 2x 高分辨率推理（大图定位偏差 -43%、锐度 +95%）；Android 端永远 480×640 再放大回全图——2048px 输入下 4px 线在 DexiNed 分量里不足 1px，放大回来是糊带
- 修复：Android 权重重新生成为**动态轴版**（per-channel INT8 **非对称**重量化，480×640 ~ 960×1280 任意尺寸直推；vs 原版 fuse 偏差 mean 0.036、sigmoid 0.5 mask 一致率 99.98%）；`runDexined` 按 Python `_DEXINED_SCALE_SCENES` 同式路由：lineart/blueprint/pixel → 2x 夹 [480×640, 960×1280]，photo/sketch → 1x（弱对比 2x 感受野失配召回 -12pp 保护）

### 4. 端到端验证（2000px 合成图，模拟新旧 Android 行为）

| 指标 | 旧（384 块 + DexiNed 480×640） | 新（896 块 + DexiNed 2x） |
|---|---|---|
| 召回 | 1.0000 | 0.9931 |
| 精确率 | 0.4124 | **0.5038**（+9.1pp） |
| 背景误报均值 | 0.037074 | **0.024397**（**-34% 噪点**） |

其他：`repack_dexined_dyn.py` 重打包脚本入库（可复现）；契约测试补 Android DexiNed 动态轴断言；APK versionCode 22

## v1.25.0（2026-10-21）：Android ensemble 生效硬保障 + 模型档位全链路透明化

**背景**（用户 v1.24.2 实测反馈）：「为什么只显示 17MB 的模型，66MB 的优先使用模型呢？是否起到作用？优化 UI 让我知道是用什么模型」——三个问题对应三个根因。

### 1. ensemble 生效硬保障：assets noCompress

- **根因**：gradle 未配置 `noCompress 'onnx'` → 双权重在 APK 内被 AAPT2 **deflate 压缩存储** → `NeuralEngine.hasEnsemble()` 的 `assets.open().available() > 1MB` 检测中，压缩 asset 的 `available()` 只保证「无阻塞可读字节数」（部分 Android 版本返回内部缓冲区大小而非文件总长）→ DexiNed 检测可能**静默失败**，ensemble 降级单模且无任何报错——正是「只显示 17MB 模型」的疑似根因
- **修复**：`androidResources { noCompress 'onnx' }`——asset 走 mmap，`available()` 恒等于文件真实大小，双权重检测稳定；加载免解压更快。代价：APK 变大（onnx 不再压缩，51.8MB 权重按原样存储）
- **注意**：v1.24.x 及更早 APK 里 onnx 是压缩的，该检测行为依设备而异——**从 v1.24.x 升级请直接装 v1.25.0 覆盖（versionCode 21）**

### 2. 「17MB」文案来源：卡片说明停留在 v1.19.1

「神经先验」卡片说明文字还是 v1.19.1 单权重时代（"内置模型随 APK 打包 17MB"）——v1.23.0 加入 DexiNed 后从未更新，用户自然以为只有 17MB 模型。重写为双权重版本（线稿模型 informative 17MB + 边缘模型 DexiNed 34.8MB → ensemble 三语义融合，任一权重异常自动托底并标注）

### 3. 模型档位全链路透明化

- **就绪状态行**带模型明细：「双权重就绪：线稿模型 17MB + DexiNed 34.8MB → ensemble 三语义融合（最优档）」；托底档如实说明原因
- **转换完成最终状态**追加「神经档位：…」——此前档位提示在「推理完成→管线渲染」间一闪而过即被覆盖，用户无从得知实际用的哪个模型
- **诊断自检**神经推理行带档位（「神经推理：通过（ensemble 三语义融合（DexiNed + 线稿模型 + 梯度，最优档），256px，…ms）」）；非 ensemble 档位明确告警 ⚠ 并给出处理建议
- APK versionCode 20→21 / versionName 1.24.2→1.25.0

## v1.24.2（2026-10-21）：Android 神经推理 cast 崩溃 + 桌面 EXE 双击 GUI 崩——双平台启动级修复

**两个用户实测暴露的启动级 bug**（v1.24.1 修好 opset 后 Android 神经推理终于走到 cast 这一步暴露；桌面 EXE 双击 GUI 崩则是 v1.16.0 GUI 改版起就存在的纯 Python 导入链错误，CI 只测 CLI 路径从未覆盖双击路径）。

### Android：ORT Result cast 崩溃

- **根因**：`NeuralEngine.kt` `runDexined` 的 `output[outName] as OnnxTensor`——ORT Java API 的 `Result.get(String)` 返回 `Optional<OnnxValue>`，直接 cast 抛 `java.util.Optional cannot be cast to ai.onnxruntime.OnnxTensor`。修复为先 `.get()` 取出再 cast（`get(int)` 按索引重载直接返回 `OnnxValue`，`inferBlock` 的 `output[0]` 原写法无此 bug；v1.24.2 首推 CI 曾对它误加 `.get()` 导致 Kotlin 编译失败，已回滚）
- **暴露时序**：v1.23.0 引入 Kotlin 双引擎时 bug 即存在，但 DexiNed 先被 opset 21 挡在 session 创建（v1.24.0 报 ORT_INVALID_ARGUMENT）；v1.24.1 修好 opset 后才走到 run 输出提取处炸 cast——即 **v1.23.0 起所有版本的 Android ensemble（DexiNed 分量）从未真正跑通过**
- APK versionCode 19→20

### 桌面 EXE：双击 GUI 崩

- **根因**：`gui/preview.py` 顶层 `from ..io import is_supported_video`——但 `io/__init__.py` 为避免循环导入刻意不导出该符号（定义在 `io/video.py`）。源码态 `--gui` 同样崩，仅 CLI 路径未触及。修复为直接从 `..io.video` 子模块导入
- **PIL 缺失连环坑**：`preview.py` 显示层延迟 `from PIL import Image, ImageTk`，而 CI 构建环境从未装 pillow → PyInstaller 不收集 → EXE GUI 窗口能起但一显示预览就崩。CI 两平台构建依赖补 `pillow`
- **门禁强化**：selfcheck `_check_gui` 此前只探测 tkinter/PIL 可用性（optional 项不计入退出码）——升级为真正加载 `lineart.gui.preview` 模块，且 CI 显式断言 `optional:gui` 为 PASS（Windows）／非 FAIL（Linux，tcl/tk 收集平台差异允许 SKIP）。新增 mock 依赖测试覆盖导入链正反两向

## v1.24.1（2026-10-21）：Android ensemble 修复——DexiNed 权重 opset 21 降级 17

**根因**（v1.24.0 APK 用户实测自检失败）：Android 内置 DexiNed 重量化权重 stamp 了 opset 21——onnxruntime-android 1.17 官方支持上限为 opset 20，加载即抛 `ORT_INVALID_ARGUMENT`（"Opset 21 is under development... Current official support for domain ai.onnx is till opset 20"），DexiNed session 建不起来，ensemble 降级、神经推理自检失败。桌面 ORT 1.31 支持 opset 23 故此前 CI 与本地均未暴露。

- **Android assets dexined.onnx opset 21→17**：重量化版算子集（Conv/ConvTranspose/MaxPool/Relu/Reshape/Concat/Mul/Add + per-channel DQ(axis)）opset 13 即满足，17 留裕量；输出与 opset 21 版**逐位一致**（max|diff| = 0.0，opset_import 不参与计算图）
- **APK versionCode 18→19 / versionName 1.19.1→1.24.1**：v1.20.0 起未递增导致覆盖安装被拒，用户无法原地升级——本次修复后老 APK 可直接覆盖升级到 ensemble 可用版
- **新增契约测试** `test_android_dexined_weight_android_ort_compatible`：Android 权重 opset ≤ 20、无 block_size 属性（ORT 1.17 不支持 blockwise DQ）、无 1D bias DQ（ORT 1.31 bug）——防止未来重量化再引入 opset 21 回归
- 仓库根 `models/dexined.onnx`（桌面 EXE 用）保持 opset 21：2D blockwise DQ 的 `block_size` 属性在 ONNX 标准中 opset 21 才正式定义，降级反而不合法；桌面 ORT 1.31 完整支持

## v1.24.0（2026-10-21）：桌面 EXE 双 fp32 权重内置——神经能力完全释放

**背景**：用户指令「EXE 打包需要极致性能，两个权重都完全释放」——桌面端无移动端 5GB 内存约束，双权重用 fp32 原版直接打包，EXE 下载即用最优 ensemble 档。

### 双权重入仓库 + EXE 打包

- **仓库根 `models/` 随附双权重**：informative-drawings fp32（17MB，与 Android APK 内置版逐位一致）+ DexiNed fp32 权重「bias 修复 + 动态轴」终态（45MB，weight blockwise DQ 保留——无量化无降档，重量化 INT8 仅 Android APK 采用）。权重查找链第 3 条（仓库根 models/）在源码运行与 EXE 打包态（`_MEIPASS` 根）同构命中；env / `~/.lineart/` / `--neural-model` 覆盖机制不变
- **入库形态为图手术终态的原因**（v1.24.0 首推 CI 实测失败教训）：原版权重的 1D bias blockwise DequantizeLinear 在 ORT 1.31 抛 "block_size must be 0"——运行时图手术修复需要 onnx 包，而 EXE 不带 onnx；预修终态使 EXE 零图手术依赖开箱即跑，`--dexined-scale 2` 动态轴同步内置（`_dexined_dynamic_model` 补已动态形态幂等判断）
- **`lineart.spec`**：`datas=[("models", "models")]` 权重打进单文件 EXE；`hiddenimports` 补 `onnxruntime`（代码为函数级动态 import，静态收集兜底）确保 ORT DLL 随包——此前 CI 未装 ORT，EXE 内神经语义完全不可用
- **体积代价**：单文件 EXE 显著增大（62MB 权重 + ORT 运行时）。供应链安全不变式保留：无模型下载按钮、运行时零网络请求，权重随仓库/产物分发可审计
- **CI 门禁升级**（Windows/Linux 双平台）：`pip install` 补 onnxruntime；新增「Bundled ensemble gate」——打包态真实跑 `--semantic ensemble` 转换并校验输出，证明 EXE 内权重 + ORT 开箱即用

### 契约测试

- `test_exe_bundled_double_fp32_weights_in_repo`：双权重在仓库、体积契约（17MB/45MB 量级）、informative 全 fp32 无 FLOAT16（Cast 内存炸弹教训）、DexiNed 保留 DequantizeLinear（blockwise DQ 原版结构，防误换量化形态）
- `test_exe_repo_weights_resolve_without_env`：隔离 env 与 HOME 后双权重解析到仓库 models/（查找链第 3 条契约）

## v1.23.0（2026-10-21）：Android 双权重 ensemble 默认 + 全局归一化统计图精度升级

**背景**：用户确认移动端 5GB 可用内存、转换时间不敏感——按「时间换精度」原则重排神经链全部速度妥协点，Android 端双权重全量打包。

### Android：双权重 + ensemble 默认 + 托底链

- **APK 双权重打包**：informative-drawings fp32（17MB）+ DexiNed per-channel INT8 重量化版（34.8MB）。重量化原因：原版权重 blockwise DQ 在 Android ORT 1.17 支持不确定，Python 侧全反量化后 per-channel INT8 重量化为标准 DQ（任意 ORT 版本可跑）；与原版输出偏差 mean 0.000137、0.5 阈值 mask 一致率 99.99%
- **NeuralEngine 重构**：双槽 session 缓存（ensemble 交替调用不重建）；`runDexined`（480×640 → BGR 均值减法 → block_cat → sigmoid → 双线性回原尺寸）；`runEnsemble` 三语义 noisy-OR（与 Python `ensemble_semantic` 同式：dexided + informative 分块 + Sobel 梯度全分辨率）
- **默认 ensemble、权重异常自动托底**（pickMode）：双权重齐备→ENSEMBLE；DexiNed 缺→INFORMATIVE；informative 缺→DEXINED；均缺→传统语义。实际档位 `lastMode` 状态栏透明告知（OOM 兜底降块重跑时如实标记 INFORMATIVE）
- 状态栏文案：双权重就绪「ensemble 三语义融合（最优，开箱即用）」；推理完成带实际档位

### 桌面：全局归一化统计图 DS_SIZE 640→896

- 时间换精度实验（1200² 弱响应合成，vs 整图推理真值）：DS=640 IoU 0.7805 / 768 0.8236 / 896 0.8625（+8.2pp）/ 960 0.8819 / 1100 0.9449
- 取 896 平衡档：960/1100 收益更高但统计推理峰值逼近 4GB 预算（本机 2 核测试进程实测 OOM）——OOM 是崩溃约束非时间约束
- ensemble 梯度分量移除 1024 下采样：实测全分辨率 Sobel 反而更快（5ms vs 18ms，下采样+回放本身有开销）且无精度差（召回 0.975 = 0.975）——原下采样是无效速度优化，删除
- 次要实验否决记录：photo 场景 ensemble scale=2 召回 0.992→0.994（+0.2pp 噪声级，神经分量已补到 0.99 天花板）；render_ss=4 定位偏差与 ss=2 完全相同（渲染瓶颈不在超采样）

## v1.22.0（2026-10-21）：ensemble 三语义融合 + 神经语义包重构——细节缺失诊断与数据集验证闭环

**背景**：用户反馈细节缺失模糊，三路线推进——C 诊断（定位原因）→ A 融合（补偿短板）→ B 数据集（2000 张真实数据验证），全部结论以实测数据支撑并沉淀进代码结构与文档。

### C：细节缺失原因诊断

- **主因：neural 响应值塌缩**——informative-drawings 对弱/浅对比细节的 ink 响应集中 <0.5，被 0.5 阈值截断（阈值 0.3→0.5 细线召回 0.41→0.27 陡降；vs DexiNed 仅降 0.10）
- **次因：分辨率塌缩**——neural 任意输入缩到 512 评估，细线像素混合稀释
- 后处理滞回/渲染无损耗（滞回反而增益 0.827→0.933）；场景路由验证正确（photo 2x 召回 0.787 < 1x 0.827）

### A：ensemble 三语义 noisy-OR 融合（`--semantic ensemble`）

- 融合式：`ink = 1 - (1-dexined)(1-neural)(1-Sobel梯度)`——任一分量响应可透传、弱响应叠加过阈，补偿 neural 塌缩
- 实测（vs dexined 单语义端到端）：弱对比山脊召回 0.919→**0.992**（+7.3pp）、极浅对比 1px 细线召回 0.858→**0.975**（+11.7pp）、纹理噪线可控（端到端墨线仅 +17~32%）
- 管线/CLI/params 接入，共用 dexined_scale 场景路由；需双权重齐备

### B：2000 张公开数据集验证

- 数据集：COCO val2017 照片 500 / anime_wallpapers 插画 500 / Anime-LineArt 线稿 500 / photo-sketch-pair-500 配对 500
- ensemble 全类型召回最优：photo +9.9pp / lineart +7.4pp / illustration +3.2pp
- **神经链调参无效性实证**：semantic=dexined/ensemble 下 line_sensitivity/denoise/thickness 对输出零影响（多组参数输出逐位一致）→ README 参数表与「神经链调参说明」明示

### 结构重构：neural 单文件 → 神经语义包（本版本）

- `semantics/neural.py`（655 行，三模型混载）→ `semantics/neural/` 包四模块：
  `informative.py`（informative-drawings + 分块/全局归一化推理）、`dexined.py`（DexiNed + 两层图手术）、`ensemble.py`（noisy-OR 融合 + 梯度语义）、`postprocess.py`（共用后处理链 + 语义图加载）
- `lineart.semantics.neural` 导入路径不变（`__init__.py` re-export 全部公共接口与测试引用的内部名），pipeline/CLI/GUI/测试零改动
- 全部模块 docstring 按研究成果重写（选型依据：召回排序 ensemble > dexined > neural；定位精度排序 dexined(2x) > dexined > ensemble ≈ neural）
- 测试：新增 TestEnsembleSemantic 5 项（推理/管线/细线召回/山脊召回/噪线控制契约）；monkeypatch 目标适配拆包；全量 133 项通过

## v1.21.0（2026-10-21）：DexiNed 高分辨率推理——动态图手术 + 场景自适应分辨率路由

**背景**：神经语义结构优化专项调研（fp16/图优化/线程/插值等技术栈转换全部实测无收益）后确认：唯一有效的精度提升在结构层——DexiNed ONNX 固定 480×640 输入是精度瓶颈（resize 模糊大图细节）。验证 DexiNed 为纯全卷积（图内 Reshape 硬编码均为权重形状、ConvTranspose 无 output_shape 硬编码），图手术将输入/输出 dim 动态化后任意分辨率可直推。

### 核心改动

- **`neural.py`**：新增 `_dexined_dynamic_model()`（图手术动态化，缓存 `.dyn.onnx`，onnx 包不可用返回 None）；`dexined_semantic()` 增加 `scale` 参数（1=训练分辨率 480×640 向后兼容；2=高分辨率推理，分辨率 = 2x 原图并夹在 [480×640, 960×1280]——小图防低于训练分辨率精度下降、大图防 OOM，960×1280 峰值内存 ~2.7GB）；动态图 session 独立缓存。
- **`params.py`**：新增 `dexined_scale` 字段（"auto"|"1"|"2"，默认 "auto"）。
- **`pipeline.py`**：`_apply_scene` 返回场景名；新增 `_dexined_scale_for()` 场景路由——清晰硬边缘场景（lineart/blueprint/pixel）→ 2x，弱对比软边缘场景（photo/sketch/off）→ 1x；显式 dexined_scale 覆盖 auto。
- **`__main__.py`**：CLI `--dexined-scale auto|1|2`。

### 精度实测

| 场景/指标 | scale=1（v1.20.0） | scale=2 | 变化 |
|---|---|---|---|
| 960×1280 大圆定位偏差 ↓ | 0.3798px | 0.2147px | **-43%** |
| 960×1280 线条锐度 ↑ | 0.3865 | 0.7558 | **+95%** |
| 1920×2560 超大圆定位偏差 ↓ | 0.7510px | 0.4024px | **-46%** |
| 弱对比山脊召回（photo 2x 图）↑ | 0.874 | 0.750 | -12.4pp（路由回 1x 规避） |
| 推理耗时（960×1280） | 2.9s | 10.5s | 3.6x 慢 |

技术栈转换否决记录（全部实测）：fp16 CPU 慢 12% 且数值漂移 0.041；图优化等级默认已全开（逐位一致）；2 核线程已用满；插值方法差异 <5%。

### 兼容性

- `dexined_semantic()` 默认 scale=1：直接调用方行为不变
- 管线默认 dexined_scale="auto"：photo/sketch 场景路由 1x，输出与 v1.20.0 逐位一致（回归测试锁定）；lineart/blueprint/pixel 场景升 2x（预期精度提升）
- 1920×2560 全分辨率直推 OOM（实验验证）→ 封顶 960×1280 为设计上限
- 测试：新增 `TestDexiNedScale` 7 项（路由纯函数/非法值/推理契约/大圆精度/管线契约/photo 向后兼容逐位一致），全量 128 项通过

## v1.20.0（2026-10-20）：精度优化三连——全局归一化分块 / NMS 8向+软共识 / DexiNed INT8 量化语义

**背景**：v1.19.1 后的精度瓶颈分析确认四个方向：神经分块 InstanceNorm 统计偏差（IoU 0.8023 天花板）、NMS 4向方向量化、共识滤波硬阈值跳变、模型选型。本版本按方向 A→D 逐项落地。

### 方向A：全局归一化分块推理（`a73c3e0`）

- **根因**：InstanceNorm 用块内统计量 vs 整图统计量的系统性偏差，是分块精度唯一损失源（上界实验：注入整图统计量 IoU 达 0.9935）
- **方案**：降采样到 640 提取 per-channel 全局 mean/var → ONNX 图手术把 InstanceNormalization 替换为 Mul+Add 固定仿射变换 → 分块推理
- **实测**：弱响应合成 1200² 大图 IoU **0.8023→0.8939**（+9.2pp）；需 onnx 包，未安装回退原始分块
- 开销：一次 640² 统计推理（~0.1s）+ InferenceSession 创建（~0.5s）

### 方向B：NMS 8向 + 软加权共识滤波（`394c395`）

- **NMS 4向→8向**：对角方向细分 22.5°/67.5°，亚对角桶用双线性插值邻域比较，减少方向量化误差
- **软共识滤波**：硬阈值 0.2 的不连续跳变改为 smoothstep 软过渡带 [0.15, 0.25]，消除相邻点 dev 0.19/0.21 的亚像素抖动
- **实测**：合成圆精度——噪声场景 std 0.2883→**0.2126**（改善 26%），clean 场景 std 0.1297→0.1188
- 回退项：神经引导梯度精化（子项3）实测与双脊线配对根本冲突（弧带命中 57.2%<65%），撤销；SSR 细线抗锯齿与 1x 等价，不启用

### 方向C/D：社区量化模型调研 → DexiNed 集成

- **BiRefNet**（4.3k★）：二分分割任务（前景/背景 mask），与线稿提取不匹配，否决
- **PIDINET**（626★）：边缘检测可行但仅 PyTorch 权重，留待后续
- **DexiNed**：OpenCV 官方 INT8 量化版（45MB，BIPED SOTA）**集成落地**：
  - 新语义选项 `semantic="dexined"`（CLI `--semantic dexined`）
  - **精度实测显著高于 neural**：合成圆原始输出 std 0.3756→0.0837（4.5x 更稳）、端到端定位偏差 -0.075→**-0.042px**（改善 44%）、弱对比山脊召回 80.6%→**91.9%**
  - CPU 推理 480×640 约 2.7s（neural 2.0s，秒级可接受）；输入 resize 到训练分辨率（实测低于此精度下降：192 直推 std 0.155 vs resize 0.084）
  - **onnxruntime 1.31 兼容性自动修复**：原版权重的 bias blockwise 量化触发 ORT bug（"block_size must be 0"），首次加载图手术反量化 bias 并缓存 `.fixed.onnx`（INT8 权重保留，与全 fp32 输出逐位一致；需 onnx 包）
  - 权重寻址：`LINEART_DEXINED_MODEL` 环境变量 / `~/.lineart/dexined.onnx` / 仓库 `models/dexined.onnx`；来源 hf-mirror.com/opencv/edge_detection_dexined

### 兼容性

- 全部 API 向后兼容：默认参数行为不变（fdog 语义、NMS 8向仅重排 keep 判定）、`semantic="neural"` 路径不变
- 全量 121 项测试通过（新增 4 项 DexiNed 契约测试 + 1 项全局归一化分块精度测试）

## v1.19.1（2026-10-20）：大图分块推理 + 混合派发精度取舍

**背景**：v1.19.0 内置 fp16 模型在 CPU EP 上触发 Cast 往返内存炸弹（实测 512×384 需 1.5GB、1024×768 直接 OOM SIGKILL），用户反馈「大图转换 30 秒后闪退出 APP」。本版本修复闪退，并针对大图精度做取舍。

### 修复：fp16 否决 + 内置 fp32 原版权重

- **根因**：CPU EP 无 fp16 原生 kernel，图优化器为每个算子插入 Cast(fp16↔fp32) 往返，激活副本物化使内存达 fp32 的 ~6 倍（实测数据：512×384 需 1566MB / 768×576 需 4490MB / 1024×768 直接 SIGKILL）
- **方案**：内置模型换回 fp32 原版 17MB（`assets/models/lineart.onnx`），删除 `lineart_fp16.onnx`
- **内存系数实测**：fp32 激活 ~1.55KB/px（512×384→238MB、1024×768→1157MB、2048×1536→4685MB）

### 分块推理 + 混合派发

- **Python 端** `lineart/semantics/neural.py`：
  - 按内存预算（桌面 4GB）派发：小图（≤~1600px 长边）→ 整图推理 **逐位不变**；大图 → 640 块分块推理
  - 640 块分块精度：mask IoU **0.8810**（实测弱响应场景天花板，InstanceNorm 块内统计系统性偏差）
- **Kotlin 端** `NeuralEngine.kt`：重写为 tiling 实现（`runTiled`/`inferBlock`，OOM 时降到 256/48 重试）
- **AndroidManifest.xml** 加 `android:largeHeap="true"`
- **diagnose()** 补齐 Kotlin 神经推理自检（256×256 探针图真实跑一遍）

### 精度取舍说明

| 场景 | 整图推理 | 640 块分块 | 说明 |
|---|---|---|---|
| 粗线照片（1536×2048） | — | **IoU 0.9999** | 粗线响应强，分块精度几乎无损 |
| 弱响应合成（460×520） | 参考 | **IoU 0.8810** | InstanceNorm 块内统计偏差 |

- **上界实验**：注入整图统计量 + ctx-direct 推理 = **IoU 1.0000（逐位一致）**——证明精度损失来自统计量获取，非分块结构本身
- **结论**：0.8810 远好于 v1.19.0 必闪退；小图（内存预算内）走整图路径逐位不变
- **测试修正**：`test_tiled_inference_precision` 阈值放宽到 0.88；`test_neural_semantic_tiled_path_e2e` 修正粗线双脊响应判定带设计缺陷（±4px 带恰落在双脊中心谷底，整图参考也仅 36%）

### 验证

- 19 项神经测试全绿（含 2 项分块精度契约）；fp16 打包检查 → fp32 打包 + dtype 断言（防回归）
- perf_v13 基线逐位一致；60 项压测全过

## v1.19.0（2026-10-20）：Android 内置 fp16 模型开箱即用 + 神经先验默认开启

### 内置 fp16 模型：APK 打包小模型，打开即用

- **为什么**：v1.18.0 的 Android 神经先验要求用户 SAF 手动导入 17MB 权重，「有 UI 无权重」对普通用户仍是门槛；内置完整 fp32（17MB）又过分膨胀 APK
- **方案**：权重 fp32→fp16 存储转换（onnxconverter-common，`keep_io_types=True`），**8.2MB 打进 assets**（`assets/models/lineart_fp16.onnx`，APK 51.4→约 60MB）；CPU EP 上 fp16 算子经 Cast 回 fp32 执行，不依赖设备 fp16 kernel，onnxruntime-android 完全兼容
- **精度选型实测**（三场景：角/弧/交叉 σ0.8、拍照级模糊 σ2.0、photo 渐变+纹理；「精准优先」标准）：
  - **fp16**：ink MAE ≤0.00021、端到端 IoU 0.988~0.999、photo 结构保留 99.9%——近乎无损，**选型采用**
  - int8 静态量化（QDQ per-channel，opset 升 17 后 4.18MB）：ink MAE 高 12.6 倍（0.00265）、端到端 IoU 仅 0.924~0.964、photo 结构保留 93.4%（丢 6.6%）——**否决**；且 QDQ int8 在 onnxruntime-android CPU EP 的内核覆盖无法本地验证，校准集泛化风险高
- **模型优先级**：用户 SAF 导入的 fp32 完整版（`filesDir/lineart.onnx`）存在时优先于内置 fp16；状态行三态显示（导入 fp32 / 内置 fp16 / 缺失）
- **供应链**：内置模型由 hf-mirror 的 rocca/informative-drawings-line-art-onnx fp32 权重本地转换（转换与精度实验数据已归档），来源可复现；CLI/Python 侧供应链策略不变（仍不自动下载、不进 Python 发行包）

### 神经先验默认开启：先定位所有细节、再转白底黑线

- 内置模型就绪时神经语义开关**默认开启**（打开 APP 即可用，无需任何导入操作）
- 转换流程语义：每张图先经 Kotlin 侧 ONNX 推理**定位全部线条细节**（产出 ink 置信图），再进入管线神经后处理链渲染白底黑线（精度优先路线，非速度优先）
- 桥接契约复验（模拟 Kotlin fp16 路径 → `to_semantic_map` → `convert_file`）：三场景端到端 IoU 0.987~0.992，弧线命中 95.0%/78.9% 与 fp32 完全一致

### 验证

- 114 项测试全绿（新增 2 项：内置 fp16 模型打包存在性 CI 可跑 + fp16/fp32 精度契约 IoU ≥0.98/ink MAE ≤0.001）；perf_v13 基线逐位一致（0.0615px/散布 0）；60 项压测全过

## v1.18.0（2026-10-20）：神经链弧线/交叉修复 + Android 神经先验落地

### 神经后处理链重构：转角/弧线/交叉模糊修复

- **根因**（8px 粗弧线竖直扫描实测）：神经对粗线（≥5px）的响应是**内外边缘双峰、中心谷底趋零**（峰 0.34/0.41、谷 0.001），滞回输出天然是双脊线——与 fdog 粗线同构；而 v1.17.0 的 EDT 骨架化在 2px 斜双线上断续（弧带命中 98%→85%）、16px 断点桥接配对又删点（→80%），且 2px 双线本身不可再骨架化（Zhang-Suen 的 p4·p6·p8 条件保护最小稳定结构，EDT dist<2 直通）
- **修复**：移除骨架化与断点桥接，双脊线直接交渲染端 `pair_edge_mask` 配对合并取中点（fdog 粗线的既有设计路径），另补小实心块清理（外接框 ≤5，对齐旧链骨架化缩面积后的清理语义）
- **效果**（角/弧/交叉合成场景，±2px 严格容差）：σ0.8 弧线连续性 **58%→94%**、σ2.0 拍照级模糊 **28%→79%**，模糊场景拐角 3/4→4/4、交叉中心白洞消除；photo 场景结构保留 +41%（ink 1423→2002 点）；耗时持平（1.03s vs 1.05s）
- **兼容**：`--semantic-map` 外部语义图路径同步受益并**升级为整链共享**（v1.17.0 仅共享后处理与渲染配置，gray 仍走光照归一/保边去噪预处理链，软化过渡带经 SSR 的 FWHM 宽度展开把 2px 线渲染成 6px、与直连 neural 的 IoU 仅 0.614；现与 neural 同步跳过，IoU **0.614→0.99**）；fdog 默认路径零改动，perf_v13 精度基线逐位一致（0.0615px/散布 0）

### Android 神经先验卡片：从「外部语义图」到「应用内推理」

- **为什么**：v1.17.0 只落地了 CLI/API 的 `semantic="neural"`，APK 的「神经先验」卡片仍是 v1.14.0 的外部语义图选择器（要求用户自行跑 Anime2Sketch/DexiNed），能力与入口脱节
- **方案**：Chaquopy 预编译索引无 onnxruntime（实测 404），改用官方 **onnxruntime-android**（Java API）在 Kotlin 侧完成 informative-drawings 推理（`NeuralEngine.kt`：Bitmap→NCHW RGB float→ink PNG），输出经 `convert_file` 的 `semantic_map_path` 走管线神经链（v1.17.0 起 `neural_source` 判定共享后处理与渲染配置），bridge 零改动
- **权重策略**：APK 不内置 17MB 权重（对齐 CLI 不自动下载的供应链安全策略），用户经 SAF「导入权重」选择 `lineart.onnx` 拷贝至 `filesDir`；未导入时开关回弹并提示来源
- **验证**：Kotlin 推理路径桥接模拟（Python onnxruntime 产出同格式 ink 图→`to_semantic_map`→`convert_file`）弧线命中 95%；`test_semantic_map_path_shares_neural_chain` 契约测试实测两路径 IoU **0.99**（8bit PNG 量化往返误差 0.0039、滞回 mask 层 IoU 0.996）

## v1.17.0（2026-10-20）：内置神经语义（ONNX 推理 + 专用后处理链）

### `semantic="neural"`：informative-drawings ONNX 神经先验

- **为什么**：纹理/轮廓/高光的区分是语义级问题，内建梯度语义在弱对比结构边缘上召回极低（合成场景实测：弱对比山脊 4.6%、纹理背景中轮廓 13.3%）
- **效果**（合成场景量化）：弱对比山脊召回 **4.6%→72%**，纹理背景中轮廓 **13.3%→90%**，纹理噪线与内建同级（4.7% vs 5.0%）；AI 视觉评审线稿质量 4→6.5 / 5.5→8.5 分
- **可选增强设计不变**：仓库不分发权重（供应链安全），自行放置 informative-drawings lineart ONNX（约 17MB）+ `pip install onnxruntime` 后启用；权重搜索顺序 `--neural-model` 参数 > `LINEART_NEURAL_MODEL` 环境变量 > `~/.lineart/lineart.onnx` > 仓库 `models/`；Session 按权重路径缓存
- **入口**：Python API `LineArtParams(semantic="neural")` / CLI `--semantic neural [--neural-model <path>]`；桌面 GUI / Android 的外部语义图入口继续可用

### 神经专用后处理链（外部 `--semantic-map` 同步受益）

神经图不能走内建 `_build_mask`（原图梯度 NMS），三处实测依据：
- **NMS 峰位错开**：神经置信峰在线中心、原图梯度峰在线边缘，1~4px 神经拖尾下二者交集仅 11%，强联合会砍掉大部分召回
- **梯度门控放大噪线**：`verify_with_gradient` 的增益门控（gain=6+膨胀）把神经图纹理弱响应放大成噪线（草地场景 3.98%→7.38% 反而恶化）
- **预处理抹掉弱边缘**：光照归一化/保边去噪把弱对比山脊带响应 p90 从 0.34 抹到 0.004——神经推理改在预处理链之前执行

定稿链路：**对比度拉伸(0.2/0.4) → 滞回直通(low_ratio=0.60) → 闭运算(3) → 骨架化 → 断点桥接(16px) → 小域清理**；渲染端墨度 ×2.2 补偿（弱响应 0.34 默认厚度下成不可见浅灰线）+ 厚度保底 1.2px（1px 骨架线 splat 稀释）

### 权重实验记录（contour 风格否决）

同期实测 contour 风格权重（xmanifold 版）：草地纹理响应 p90=0.973（全响应，比 lineart 风格更差）、photo 场景丢弱边缘（r1 带 p50≈0），故不采用

### 验证

- 108 项测试全绿（新增 `test_neural_semantic.py` 11 项：后处理链 6 项无权重依赖 CI 可跑 + 权重缺失报错 2 项 + 推理/管线/超越性契约 3 项有权重时执行）
- perf_v13 精度基线逐位一致（0.0615px / 散布 0），压测 60 项全过，selfcheck 核心 11 项全绿

## v1.16.1（2026-10-20）：统一线宽 + SSR 渲染方向修复

### 统一线宽（CAD 无级差线稿）

- **`uniform_width` 新参数**（默认关闭，完全向后兼容）：开启后粗线双边缘配对合并照常执行（防双线），但不再按原图 FWHM 线宽展开，全部按 `thickness` 统一渲染——工程图 / 制图场景的无级差线稿一步到位
- **入口全覆盖**：Python API `LineArtParams(uniform_width=True)` / CLI `--uniform-width` / Android 参数区「统一线宽（忽略原图粗细）」开关
- **角部断口桥接**：粗线拐角处脊线配对失败残留点与合并中线错位半线宽，细线渲染下显形为断口；新增 `_bridge_uniform_gaps()` 闭运算连接（实测连通域 26→3，厚度保持 1px 不增粗，连接段灰度 ≈0.85 浓度）

### Bug 修复：SSR 渲染方向（v1.10.0 遗留）

- `_render_splat` 中 `(nx,ny)=(-dy,dx)` 把 refine 输出的**法向**当**切向**用，`thickness>1` 时沿切向涂抹——默认模式被 FWHM 宽展开的墨海掩盖从未显形，`uniform_width`（等宽 1px）下暴露为线宽完全失效
- 修复为 `nx=-dx, ny=-dy`（与 kk 点列展开约定一致）；全量 97 测试 + perf_v13 基线逐位一致 + 压测 60 项全过，证明旧行为无既有依赖

### 验证

- 97 项测试全绿（新增 `test_uniform_width.py` 7 项契约：默认保留原宽 / 统一后塌宽 / thickness=3 输出 3px / 无双线回归 / 中线位置一致 / 默认逐位一致 / CLI 旗标解析）
- perf_v13 精度基线逐位一致（平均 |误差| 0.0615px / 散布 0.0000px），压测 60 项全过，selfcheck 核心 11 项全绿

## v1.16.0（2026-10-20）：应用图标 + 场景预设 + 底部操作栏与分享 + 第三轮算法调研

### Android UI 三项升级

- **应用图标**：告别系统默认图标——白底三弧线 + 蓝点原创设计（呼应「白底黑线」核心业务），5 密度 legacy mipmap（48~192px）+ Android 8.0+ 自适应图标（432px 透明前景 + 白色背景层），4x 超采样抗锯齿绘制
- **场景预设**：首页新增预设卡片，四个一键选项直达管线场景适配——
  - **通用**（默认）：自动检测图片类型（像素画 / 蓝图 / 手绘扫描 / 数字线稿 / 照片），推荐首选
  - **CAD**：工程图 / 矢量渲染 / 数字线稿 → 标准管线，最高几何与亚像素精度
  - **插画**：手绘 / 漫画 / 扫描稿 → Sauvola 自适应阈值语义，强化铅笔弱线检出
  - **照片**：照片 / 自然图像 → FDoG 流场语义提取主体轮廓，抑制纹理噪线
  - 预设不覆盖你手动调整过的参数（v1.13.0 覆盖语义不变），切换即时显示场景说明
- **底部常驻操作栏 + 分享**：转换 / 保存 / 分享三按钮固定屏幕底部（不再被长内容推出视野），状态文字随栏固定显示；新增分享按钮——结果图经 FileProvider 临时授权直接 ACTION_SEND 给微信 / QQ / 邮件等任意应用，无需先保存到相册

### 第三轮算法调研（详见 README 调研章节）

- 吸收 **informative-drawings**（CVPR 2022，453★）核心思想：「线稿应传达几何与语义，而非纹理」——其用 CLIP 感知损失 + 深度图几何监督训练 GAN 生成线稿
- **结构张量纹理门控实验（实现 → 实测推翻 → 撤销）**：为解决照片场景纹理残留，实现结构张量相干性门控；实测语义响应 ink>0.3 位置上：草叶纹理相干性 0.83、真实轮廓 ≈0、高光斑 0.86——判据方向完全相反（门控会保留草叶、削掉轮廓）；min_area 过滤同样无差别杀伤（草地噪线 5.93%→0.37% 的同时真实轮廓 1.71%→0.43%）。结论：**纹理 / 轮廓 / 高光的区分是语义级问题，一切传统局部特征（相干性 / 长度 / 面积）均失效，神经先验（`--semantic-map`）是正解**。实验代码已撤销，实测数据存档于 README

### 其他

- `bridge.convert_file` 新增可选 `scene` 关键字参数（末尾追加、向后兼容 v1.14/v1.15 的位置参数调用），APK 预设经此入口直达管线场景；新增 7 项 bridge 场景契约测试（90 项总）
- 管线核心零改动：83 项既有测试、perf_v13 精度基线、60 项压测全部不受影响


## v1.15.1（2026-10-20）：修复点击预览闪退 + 生活照片质量说明

**背景**：v1.15.0 APK 实测两类反馈——(1) 点击原图/结果缩略图进入大图预览时应用闪退；(2) 生活照片转白底黑线效果不理想。

### 修复（预览闪退）

- **根因**：自定义视图 `ZoomImageView` 只定义了 `(Context)` 单参构造函数，而 Android 的 `LayoutInflater` 从 XML inflate 时调用 `(Context, AttributeSet)` 构造函数，找不到直接抛异常闪退。修复：改用 `@JvmOverloads` 补齐 `(Context)` / `(Context, AttributeSet)` / `(Context, AttributeSet, Int)` 三参构造；同时移除冗余的 `supportRequestWindowFeature`（主题已全屏 NoActionBar）

### 关于「生活照片转换效果差」（算法边界说明）

- 传统信号算法（FDoG/XDoG）只依据**边缘强度**提取线条，无法区分「主体轮廓 / 内部纹理 / 高光反光」三者的**语义差异**：实测模拟生活照片（草地背景+苹果），背景纹理抑制干净、轮廓完整，但苹果高光点被误画成独立「孔洞」轮廓——这是缺乏语义理解的典型表现，不是参数 bug
- 复杂生活照片（人像/街景/自然纹理）会放大此问题：要么纹理变杂乱，要么强去噪丢细节
- **建议**：复杂照片使用「神经先验」入口（三端都支持）——用 Anime2Sketch / DexiNed 等预训练模型对同一张图生成语义响应图后加载，可显著提升照片线条连贯性与主体语义。详见 README「神经先验」


## v1.15.0（2026-10-19）：粗线双线化修复 + APP 内大图预览

**背景**：用户实测 v1.14.0 反馈两个问题——(1) 白底黑粗线图输出双线，一条线被画成两条；(2) 需要应用内查看转换结果大图。双线问题定位出三层根因，全部修复。

### 修复（双线的三层根因）

- **场景误判（主因）**：粗线图被自动检测判为「像素艺术」场景 → 该场景关闭边缘合并（merge_edges=False）并加粗渲染（thickness=4），粗线的双边缘脊线不合并直接输出双线。旧判据「块状实心率」（距边缘 >1.5px 的全局占比）对粗线失效——12px 线内部同样大片距边缘 >1.5px（实心率 0.67 > 阈值 0.5）。新判据**深核占比**：深核 = 距背景 ≥12px 的区域，对深核掩码做连通域分析（细长度 ≥8 排除超粗线芯、填充率 <0.3 排除环形芯带）。物理本质：只有大实心色块才有深核（pixel_art 样本 0.636、色块拼图 0.532、40px 独立方块 0.160），线状结构的深核只是中心细条带（各宽度粗线/环/框均 <0.001），阈值 0.15。宁可漏判（不启用像素艺术预放大，质量不变）不可误判（像素场景必双线）
- **边缘合并距离上限不足**：双边缘配对距离默认 6px 只覆盖 ~7px 线宽，8px+ 粗线配对距离超限退化为双线（v1.14.0 实测 8px+ 垂直线整段丢失或双线）。默认 6 → 20（params 默认值 + render/pipeline 函数签名三处同步）
- **暗谷判据过松（上项扩大的副作用）**：配对距离扩大后，噪声圆盘外的灰背景恰好通过 0.75×灰亮部的宽松暗谷判据，噪声点远距误配对把中点渲染进圆盘（噪声圆回归一度失败）。暗谷比 0.75 → 0.6；修复后噪声圆径向散布 0.288，还优于 v1.14.0 的 0.426，且各灰度（100/150/180）粗线单线保持

### 新增（APP 内大图预览）

- 点击主界面的原图/结果缩略图进入全屏预览：双指缩放 0.5×~32×（最近邻插值，放大看像素细节不糊）、单指拖拽（自动钳制不出界）、双击在适屏/1:1 像素级间切换，顶部信息条实时显示尺寸与缩放倍率，黑底沉浸主题看线稿对比度最佳
- 主界面缩略图为 1080 降采样，预览加载全分辨率（最长边 2048）结果原图，对齐桌面 GUI 的大图查看能力

### 测试

- 新增双线化回归测试 10 项（直线各宽度 2-20px 的单线/断裂双检、圆各宽度 2 段、粗空心框四边单线、噪声圆鲁棒、灰色粗线暗谷边界、场景防误判与防误伤），总数 73 → 83
- perf_v13 精度基线逐位一致（细线 0.0615px / 模糊边 0.2256px / 圆半径 0.2605px）、60 项压测全过、selfcheck 核心项全绿

### 已知边界

- 线端点/拐角处存在 1px 级毛刺（v1.14.0 既有渲染层边界行为，非本版回归，本版整体更少）；`--render-mode shape` 可缓解


## v1.14.0（2026-10-19）：Android 可用性修复 + UI 重设计 + 神经先验入口

**背景**：用户实测 v1.13.0 APK 反馈三类问题——手机上无法完成转换、界面配色差（字体与背景对比不足）、README 介绍提到模型但软件内毫无体现。全部定位修复。

### 修复（Android 无法转换的三个根因）

- **HEIC 手机照片必失败**（主因）：手机相机默认 HEIC 格式，Chaquopy 锁定的 opencv 4.5.1 无 HEIF 编解码器，`imdecode` 返回空直接抛「无法读取图像」。输入图改由 Kotlin 端 **Android 原生解码**（API 28+ `ImageDecoder` 原生支持 HEIC 且自动应用 EXIF 旋转；API 24-27 `BitmapFactory` + 手动 EXIF 方向修正），统一存 PNG 后交给 Python，同时修好 JPEG 照片方向颠倒问题
- **「纯黑白」开关失效**：桥接层误用 `output="binary"`（灰度输出的历史既有行为），纯黑白实际由 `mode="binary"` 控制（v1.12.0 确认的契约），已更正
- **大图 OOM 风险**：APK 侧 `max_side` 由 4096 收紧到 2048（Android 应用堆约 256MB，4096² 输入的多份 float32 中间缓冲极易溢出；2048 对手机预览与保存精度已足够）

### Android UI 重设计（对比度崩溃修复）

原主题 DayNight + 强制白背景，系统暗色模式下文字自动变浅而背景仍白，对比度崩溃。改为**固定浅色主题 + 全控件显式配色**（正文 #1A1A1A on 背景 #F4F5F7，对比度约 15:1，WCAG AAA）；布局重做为卡片式（白色圆角卡片 + 边框）：前后对比预览、参数滑动条带实时数值、转换/保存主按钮、状态文字按结果着色（成功绿 / 失败红）。

### 神经先验入口（「介绍有模型但软件没有」）

本项目**不依赖任何模型权重、完全离线可用**——没有模型下载按钮是设计使然。README 提到的 Anime2Sketch / DexiNed 是**可选增强**，本版本把入口补齐到全部三端：CLI `--semantic-map`（既有）、桌面 GUI「加载神经先验图（可选）」按钮、Android「神经先验（可选增强）」卡片，并新增 **「运行诊断自检」**入口（应用内一键跑 bridge 自检，5 项端到端真实转换）。

### 修复（神经先验方向反转，新单测捕获）

`bridge.to_semantic_map` 首版把管线 ink 约定（[0,1]，1=线）做反了 180°：白底黑线未反相，背景被当作线、线被当作背景，语义先验会反向引导。已修正并同步修正 `semantics/neural.py` 误导性 docstring；桌面 GUI 语义先验入口共用 `load_semantic_map`。

### CI 门禁增强（v1.13.0 教训闭环）

android job 此前只校验 APK 内包文件**存在性**，从不执行——「构建成功但转换报错」无法拦截。新增 **bridge 真实转换冒烟门禁**：从 app.imy 提取应用代码，在 runner 上真实执行 `convert_file`（默认 + binary 纯黑白契约）与 `selfcheck`。

### 测试

新增 `tests/test_bridge.py` 9 项（bridge 契约锁定：签名/尺寸不变/binary 纯黑白/语义先验路径/线条检出、`to_semantic_map` 白底黑线与黑底白线方向、值域、selfcheck）；全量 73 项 + 20 子测试、perf_v13 逐位一致、60 项压测、selfcheck 14 项全过。

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
