# lineart-precise v1.16.0

高精度「图片 → 白底黑线」线稿转换器：亚像素级线条定位、场景自适应、外形保真双渲染路线。本版本起提供免安装桌面版与 Android 版。

## 下载

| 文件 | 平台 | 说明 |
|---|---|---|
| `lineart-1.16.0-windows-x64.exe` | Windows 10/11 x64 | 免安装单文件；双击进 GUI，命令行当 CLI |
| `lineart-1.16.0-android-arm64.apk` | Android 8.0+（arm64） | 选图 → 调参 → 转换 → 保存，管线与桌面版同源 |
| `lineart-1.16.0-linux-x64` | Linux x86-64（glibc 2.28+） | 免安装单文件 CLI |

## 系统要求

- **Windows**：Win10+ x64，无需安装 Python 或任何依赖（单文件含 Python + OpenCV + NumPy 全套运行时，v1.11.6 实测：Windows 64.5MB / Linux 99.3MB）
- **Android**：8.0+（API 24）、arm64-v8a 架构，APK 约 35MB（内含 Python 运行时与 OpenCV）
- **Linux**：glibc 2.28+（Ubuntu 20.04+ / CentOS 8+ / Debian 11+）

## 安装步骤

### Windows

1. 下载 `lineart-1.16.0-windows-x64.exe`，放到任意目录
2. **图形界面**：直接双击，进入 12 参数滑动条的实时预览 GUI
3. **命令行**：
   ```bat
   lineart.exe photo.jpg -o lineart.png
   lineart.exe ./photos -o ./results --batch --workers 4
   ```
4. 首次运行若弹出 SmartScreen 譝告（未购买代码签名证书的正常提示）：点「更多信息」→「仍要运行」

### Android

1. 下载 `lineart-1.16.0-android-arm64.apk`，用文件管理器打开
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
