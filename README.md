# lineart-precise

自研**高精度「图片 → 白底黑线」转换器**。面向线稿提取 / 上色底稿 / 印刷制版等对线条位置精度苛刻的场景，纯 OpenCV + NumPy 实现，无需下载深度学习模型权重，离线可用。

## 全网方案调研（2026-10，两轮深挖：主流 + 小众）

### 深度学习路线（生成式"自动画图"）
| 方案 | 星数 | 说明 | 未直接采用原因 |
|---|---|---|---|
| carolineec/informative-drawings (CVPR'22) | — | 无配对 GAN 线稿生成，视觉风格最佳 | 需 GPU+权重，线条位置不可控 |
| Mukosame/Anime2Sketch | 2135★ | 对抗域适应素描提取，动漫/插画佳 | 需权重；已做成可选语义先验接口 |
| xavysp/DexiNed | 865★ | BIPED SOTA 边缘检测网络 | 需权重；已做成可选语义先验接口 |
| hellozhuo/pidinet (ICCV'21) | 625★ | 像素差分轻量边缘网络 | 同上 |
| MengyangPu/EDTER (CVPR'22) | 325★ | Transformer 边缘检测 | 同上 |
| bobbens/sketch_simplification | 754★ | "Learning to Simplify" 草图简化 | 方向相反（粗稿→净稿），借鉴其保边思想 |

### 传统路线（本项目同赛道，重点研究小众项目）
| 方案 | 星数 | 核心思想 | 吸收情况 |
|---|---|---|---|
| SSARCandy/Coherent-Line-Drawing | 572★ | Kang et al. ETF 流场 + FDoG 流线 DoG | **已吸收**：`lineart/flow.py` 全向量化重实现 |
| esimov/colidr | 55★ | Go 版 Coherent Line Drawing | 算法参考 |
| uva-graphics/coherent_line_drawing | 17★ | Kang 论文官方独立程序 | ETF 权重设计参考 |
| abadi199/adaptive-dog | 5★ | 自适应 DoG | 参数自适应思想 → 噪声自适应门限 |
| 0xffffffffun/coherent-line-drawing | 3★ | Python 简化实现 | 对照验证 |
| ishanrai05 / sinAshish FCNN 简化 | 39★/33★ | PyTorch 草图净化 | — |
| amaldevp/RoughSketchSimplification | 12★ | Delaunay 多笔画简化 | — |
| exaterin/Line-art-extraction-using-XDoG | — | 单尺度 XDoG | 已超越（多尺度+梯度验证） |
| ljsabc/MangaLineExtraction_PyTorch | 200★ | 漫画结构线提取 | 面向漫画，照片不稳 |

> 检索方式：GitHub API 全量搜索 `image to sketch` / `photo to lineart` / `sketch simplification` / `coherent line drawing` / `Anime2Sketch` / `DexiNed` / `PiDiNet` / `EDTER` / `ControlNet lineart` / `CoTRefiner` / `sketchKeras` / `FDoG`，含 0★ 项目逐个人工评估。

## 架构：语义-定位解耦（精度核心）

线条**位置**与"哪里有线"的判断完全分离：
- **语义层**（哪里有线，可插拔）：`fdog` 流线连贯响应（Kang et al.）/ `xdog` 经典多尺度 / `adaptive` Sauvola 自适应阈值（拍照手绘/扫描文档）/ 外部神经先验图（Anime2Sketch、DexiNed 等，`--semantic-map`）
- **定位层**（线在哪里，精度保证）：梯度 NMS 脊线 + 亚像素抛物线拟合（0.75px 步长）+ 双线性 splat 抗锯齿渲染

### 渲染双路线（v1.10.0，`--render-mode`）

| | splat（默认）：中心线重绘 | shape：外形保真 |
|---|---|---|
| 外形来源 | 骨架化 → 1px 中心线 → FWHM/参数化宽度重新画线 | 原图墨度 50% 半深度切分（AA 边面积重建无偏） |
| 宽度渐变/笔锋 | 单值宽度模型，无法表达 | 原样保留（变宽线 corr 0.619→1.000） |
| 宽线（>merge_gap） | 边缘配对失败退化为双线（10px 线 IoU 0.199） | 原图连通找回线心（10px 线 IoU 1.000） |
| 中心位置 | 亚像素拟合（圆径向误差 0.065px） | 半深度外形（圆径向误差 0.012px） |
| 性能（4096² 压测） | 基准（3992ms） | 快约 2 倍（2000ms） |

shape 路线的角色分工：**原图为主**（外形直接取自墨度场半深度阈值），**转换为辅**（语义响应经测地重建填充宽线线心做墨度、滞回连通域剔除无响应暗斑、形态学/连通域清理）。墨度三模式：`response`（检测强度软边）/ `source`（原图笔触浓淡）/ `flat`（均匀黑）。

## 场景自适应（v1.2.0，第 6 轮：更多场景转换）

`--scene auto`（默认）自动识别输入场景并适配管线：

| 场景 | 检测特征 | 管线适配 |
|---|---|---|
| **拍照手绘草图** | 浅底+低饱和+细线+背景低频起伏（阴影） | Sauvola 自适应语义 + 弱全局归一化 |
| **扫描文档/手稿** | 装订阴影、暗角、污渍、文字行 | 同上（局部阈值天然抗阴影） |
| **蓝图** | 蓝主导色相 + 亮线 | V×(1-S) 白线提取 + 反相成白底黑线 |
| **像素艺术** | 颜色少 + 无渐变中间调（硬边） | 最近邻 4x 预放大（产生可精化过渡带）+ 缩回原尺寸输出 |
| **数字线稿/CAD** | 浅/深底 + 低饱和 + 背景平坦 | 标准管线（含黑底自动反转） |
| **照片/自然图像** | 色彩丰富或连续色调 | FDoG 高连贯 + 强去噪 + 纹理抑制 |

关键实现：
- **Sauvola 积分图二值化**（`lineart/adaptive.py`）：窗口均值/方差 O(1)/像素，软过渡输出避免硬边界干扰梯度验证层。阴影下同样深度的线响应强度一致（全局阈值做不到）
- **场景判别统计量**（`lineart/scenes.py`）：四角中值亮度/散布、背景低频不均匀度（形态学闭运算去线后估计）、线条占比（相对背景 -35 级，适配铅笔弱线灰度 120~160）、渐变中间调占比（像素艺术 <0.35 vs 抗锯齿线稿 ≥0.44，实测标定）
- **黑底 CAD 在反转前检测**：场景检测不依赖底色归一化，黑底图直接判 lineart 走标准管线

## 本轮融合升级（第 2 轮调研成果）

1. **ETF + FDoG 流线语义**（吸收 SSARCandy/Coherent-Line-Drawing 572★ 及 3 个小众实现）：切向流场迭代平滑（Ws/Wm/Wd/Φ 权重）→ 灰度沿流线平滑 → 多尺度 XDoG。大图自动半分辨率计算流场，语义图无需全分辨率。
2. **导向滤波保边去噪**（He et al.，借鉴 sketch_simplification 思想）：替代高斯预去噪，细弱线（对比度 12/255）检出率 >75% 且定位 <0.6px。
3. **EDT 中线骨架校正**：距离变换局部极大值把闭运算造成的局部粗线收敛回中线，粗线渲染（`--thickness`）以真实中线对称展开，中线偏移 <0.6px。
4. **噪声自适应门限标定**（吸收 abadi199/adaptive-dog 自适应思想）：门限系数经扫描实验标定为 1.8——弱结构边缘（对比度≈纹理 1.5 倍）覆盖率 43.9%→97.9%，强纹理区误检保持 0%。
5. **可选神经语义先验接口**：`--semantic-map` 直接喂 Anime2Sketch/DexiNed 等任一模型输出，管线负责精确定位与渲染（本机无 torch 时自动用内建语义）。

## 实测精度

### 定位精度（合成真值，192px 圆 r=40.30px）
| 方法 | 平均误差 | 误差标准差 |
|---|---|---|
| **lineart-precise** | **0.040 px** | **0.101 px** |
| OpenCV Canny | -0.037 px | 0.229 px |
| 自适应阈值 | 2.340 px | — |

### 鲁棒性（38 项自动化测试全通过）
| 场景 | 指标 |
|---|---|
| 亚像素边缘 64.0/64.25/64.5/64.75 | 误差 <0.35px |
| 5% 高斯噪声圆 | 径向误差 <0.70px，覆盖 >80% |
| 渐变光照背景 | 径向误差 <0.70px |
| 三次贝塞尔曲线 | 平均误差 <0.65px，漏检 <10% |
| 细弱线（对比度 12/255） | 行覆盖 >75%，定位 <0.6px |
| 强纹理背景（对比度 26）+ 结构圆 | 纹理误检 <2%，圆覆盖 >90% |
| 空白噪声图 | 误检 <0.5% |
| 粗线渲染 thickness=3 | 中线偏移 <0.6px |
| 弱结构（对比度≈纹理1.5倍） | 覆盖 97.9%（noise_k=1.8 标定值） |
| 3px 硬边粗线（合并后） | 虚假墨 <40px、丢线 <5%、单线位置逐像素吻合 |

## 下载安装包（v1.11.1+）

免安装三平台产物由 GitHub Actions 云端构建，随 [Releases](https://github.com/liuben3223380369-ops/lineart-precise/releases) 发布：

| 文件 | 平台 | 用法 |
|---|---|---|
| `lineart-*-windows-x64.exe` | Windows 10/11 x64 | 双击进 GUI；命令行 `lineart.exe in.jpg -o out.png` |
| `lineart-*-android-arm64.apk` | Android 8.0+ arm64 | 安装后选图→调参→转换→保存 |
| `lineart-*-linux-x64` | Linux x86-64 | `./lineart in.jpg -o out.png` |

桌面单文件内含 Python + OpenCV + NumPy 全套运行时（Linux 实测约 80MB），无需任何环境；Android 版由 Chaquopy 编入同源管线。安装说明与版本日志见 [RELEASE_NOTES.md](RELEASE_NOTES.md)（即 Release 页正文）。

## 安装与使用

```bash
pip install -r requirements.txt

# 单张（默认 FDoG 流线语义）
python -m lineart photo.jpg -o lineart.png

# 目录批量（多线程）+ 纯黑白 + 加粗
python -m lineart ./photos -o ./results --mode binary --thickness 2 --batch --workers 4

# 经典 XDoG 语义
python -m lineart in.jpg -o out.png --semantic xdog

# SVG 矢量输出
python -m lineart in.jpg -o out.svg --output-svg

# 外形保真路线（原图线条外形为主：宽线/变宽线/笔触保真，制版推荐）
python -m lineart in.jpg -o out.png --render-mode shape --shape-ink source

# 实时预览 GUI（12 参数滑动条）
python -m lineart --gui

# 外部神经先验（Anime2Sketch/DexiNed 输出图）
python -m lineart in.jpg -o out.png --semantic-map sketch_response.png

# API
from lineart import LineArtParams, to_lineart, to_lineart_batch
out = to_lineart(img_bgr, LineArtParams(semantic="fdog", coherence=0.7))

# 视频/GIF 逐帧处理（ETF 帧间一致性，防闪烁）
from lineart.io.video import process_video
process_video("clip.mp4", "out.mp4", LineArtParams())
```

### 参数
| 参数 | 默认 | 说明 |
|---|---|---|
| `--mode` | antialiased | 抗锯齿灰度线 / 纯黑白 |
| `--semantic` | fdog | `fdog` 流线连贯 / `xdog` 经典多尺度 / `adaptive` Sauvola 自适应阈值 |
| `--scene` | auto | 场景自适应：auto 检测 / off 关闭 / pixel·blueprint·sketch·lineart·photo 显式指定 |
| `--semantic-map` | — | 外部神经语义先验图（覆盖内建语义） |
| `--coherence` | 0.6 | 流线连贯强度 0~1 |
| `--smooth-radius` | 3 | 导向滤波保边平滑半径 |
| `--detail` | 0.6 | 细节强度 0~1 |
| `--sensitivity` | 0.5 | 线条灵敏度 0~1 |
| `--thickness` | 1.0 | 线条粗细（像素） |
| `--denoise` | 0.4 | 去噪强度 0~1 |
| `--illumination` | 0.8 | 光照归一化强度 0~1 |
| `--min-area` | 10 | 噪点连通域剔除面积 |
| `--max-side` | 4096 | 输入长边上限 |
| `--sharpen` | 0.0 | 相干冲击锐化 0~1（模糊输入推荐 0.5~0.8，只增强检测不影响定位） |
| `--render-ss` | 2 | 渲染超采样倍率（粗线宽度/边缘抗锯齿更准；pixel 场景自动禁用） |
| `--render-mode` | splat | 渲染路线：splat 中心线重绘（默认）/ shape 外形保真（原图外形为主，宽线/笔触/制版保真） |
| `--shape-ink` | response | shape 路线墨度：response 检测强度软边 / source 原图笔触浓淡 / flat 均匀黑 |
| `--shape-closing` | 1 | shape 路线外形清理闭运算尺寸（1=最小保外形，0=关闭） |
| `--consensus` | true | 亚像素偏移邻域中值共识（剔除孤立拟合噪声） |
| `--gradient-method` | farid | farid（5×5 最优导数）/ sobel |
| `--subpixel-method` | gaussian | gaussian / parabolic / zernike |
| `--gui` | off | 启动 tkinter 实时预览（滑动条调参） |
| `--batch` / `--workers` | off/4 | 目录批量处理与线程数 |
| `--output-svg` / `--simplify-tol` | off/1.0 | SVG 矢量输出与折线简化容差 |
| `bridge_gap`（API） | 8 | 断线桥接最大间隙（像素，0 关闭） |
| `adaptive_width`（API） | true | FWHM 自适应线宽（硬边细/软边粗） |
| `multiscale`（API） | false | FDoG 三尺度语义融合（0.5/0.3/0.2 加权） |

## 测试与演示

```bash
python -m pytest tests/ -q            # 38 项精度/鲁棒性/场景/shape路线测试（+15 子测试）
python demo/run_demo.py               # 全部合成场景 + 三模式对比 + 场景自适应演示
```

## 项目结构

```
lineart-precise/
├── lineart/
│   ├── __init__.py        # 公共 API + 版本
│   ├── __main__.py        # CLI（含 --gui / --batch / --output-svg）
│   ├── params.py          # 参数契约（LineArtParams 唯一来源）
│   ├── pipeline.py        # 主管线编排（语义-定位解耦 + 场景自适应 + 批量 API）
│   ├── io/                # 读写与导出
│   │   ├── __init__.py    #   图像读写（中文路径/Alpha 合成白底）
│   │   ├── svg_export.py  #   SVG 矢量输出（道格拉斯-普克简化）
│   │   └── video.py       #   视频/GIF 逐帧处理（ETF 帧间一致性）
│   ├── gui/preview.py     # tkinter 实时预览 GUI（12 参数滑动条）
│   ├── common.py          # 公共工具（双线性采样、形态学、连通域、批量 splat）
│   ├── scenes.py          # 场景自动检测与参数适配（5 场景：sketch/blueprint/pixel/lineart/photo）
│   ├── color.py           # 多色线分离与颜色保留渲染
│   ├── preprocess/        # 预处理层：输入 → 干净灰度
│   │   ├── background.py  #   底色检测与自适应反转（黑/白/灰/蓝图）
│   │   ├── illumination.py#   光照归一化（1/8 尺度背景估计）
│   │   ├── denoise.py     #   导向滤波保边去噪（自引导快速路径）
│   │   ├── text.py        #   文字区域检测与去除（MSER + inpainting）
│   │   └── shock.py       #   相干冲击锐化（值推格式，检测/定位分离）
│   ├── semantics/         # 语义层：灰度 → 线响应（哪里有线，可插拔）
│   │   ├── flow.py        #   ETF 流场 + FDoG 流线语义（Kang et al.，多尺度融合）
│   │   ├── xdog.py        #   多尺度软阈值 XDoG
│   │   ├── adaptive.py    #   Sauvola 自适应阈值（拍照手绘/扫描文档）
│   │   └── neural.py      #   可选神经语义先验接口
│   ├── localization/      # 定位层：线响应 → 亚像素线稿（线在哪里，精度保证）
│   │   ├── gradient.py    #   梯度算子（Sobel / Farid-Simon 5x5 最优导数）
│   │   ├── edges.py       #   梯度验证/NMS/滞回/骨架/连通域清理/网格去除
│   │   ├── connectivity.py#   断线桥接（骨架间隙搜索 + 端点近邻连接）
│   │   ├── refine.py      #   亚像素精化（抛物线/高斯/Zernike）+ 中值共识
│   │   ├── merge.py       #   粗线双边缘合并（脊线点对 → 线心）
│   │   ├── render.py      #   抗锯齿 splat 渲染 + SSR 超采样 + FWHM 自适应线宽
│   │   └── render_shape.py#   外形保真渲染（shape 路线：原图外形为主，检测为辅）
├── tests/test_precision.py   # 精度/鲁棒性测试
├── tests/test_scenes.py      # 场景检测/转换质量测试
├── tests/test_merge.py       # 粗线合并结构保真回归
├── benchmarks/               # 实验/基准脚本（work_scale 决策、背景估计对比、
│                             #   micro-bench、cProfile、multiscale/ETF/批量等）
├── perf_v13.py               # 六节端到端精度基准
├── stress_test.py            # 60 项压力测试
├── pyproject.toml            # 包元数据 + 依赖 + pytest 配置
└── demo/                     # 合成场景与演示
```

## 重构历史

### v1.11.1
- **修复 Windows EXE CLI 崩溃**：cp1252 控制台打印中文结果时 UnicodeEncodeError（转换已成功、仅收尾 print 崩溃）；`run_lineart.py` 启动器强制 stdout/stderr 为 UTF-8（errors=replace）
- **修复 Android CI**：移除多余 `setup-android` step（ubuntu-latest 预装 SDK，其 sdkmanager 异常），Gradle 直接用预装 SDK

### v1.11.0
- **一键打包发布**（`.github/workflows/release.yml`，tag `v*` 触发）：GitHub Actions 云端构建三平台产物并挂 GitHub Release——Windows EXE（PyInstaller 单文件，无参数双击自动进 GUI）、Linux 单文件、Android APK（`android/` 工程：Kotlin UI + Chaquopy 15.0.1 编入同源 Python 管线，numpy 1.26.4/opencv 4.8.1.78 预编译矩阵版本，arm64-v8a，演示签名环境变量注入）
- **顶层启动器 `run_lineart.py`**：解决 `python -m lineart` 在 PyInstaller 包内相对导入失败问题（spec 入口指向它）
- **发布页标配**（`RELEASE_NOTES.md`）：下载表、系统要求、安装步骤（Windows/Android）、快速上手、参数速查、已知问题、版本日志
- Android 桥接（`android/app/src/main/python/bridge.py`）：Kotlin↔Python 走文件路径，本地已验证契约（640×480 样图 276ms）；大图后台线程转换防 ANR
- 本地 Linux 打包冒烟通过：单文件 80MB（cv2/OpenBLAS/视频编解码为主）、CLI 转图 988ms；spec excludes 补 llvmlite

### v1.10.1
- **新增 `codebase-structure.md`**：项目结构总览文档（技术栈 / 目录树 / 关键模块 / 开发命令，面向 bug-fix 工作流），UTF-8 无 BOM 编码，二进制严格解码验证通过（无 U+FFFD/CRLF/BOM）
- **修正 README 失实描述**（历史核查驱动）：结构树场景数 8→**5**（comic/handwriting/architectural 经 `git log -S` 全版本核查确认从未合入 scenes.py，v1.9.0 的 D 项实际未落地，历史记录已改为如实标注）；压测数 54→**60**；测试数 26→**38**（v1.10.0 遗漏更新）

### v1.10.0
- **新增 shape 外形保真渲染路线**（`--render-mode shape`，`lineart/localization/render_shape.py`）：以原图线条外形为主、管线检测为辅的重绘路线，量化结论（benchmarks/bench_shape_route.py）：
  - 宽线外形：10px 硬边线 IoU 0.199 → **1.000**（splat 因边缘配对失败（merge_gap=6）退化为双线；shape 由原图墨度场连通性找回线心）
  - 宽度保真：2→8px 变宽线逐行宽度相关系数 0.619 → **1.000**（MAE 0.85 → 0.00px；splat 的单值宽度模型无法表达宽度渐变）
  - 中心精准度：模糊圆径向误差 0.065 → **0.012px**（半深度切分对 AA/模糊过渡带的面积重建无偏）
  - 性能：4096² 压测 3992 → **2000ms**（跳过亚像素精化与 splat 渲染两阶段）
- **外形掩膜算法**（`build_shape_mask`）：原图墨度场在滞回区域内取半深度阈值 t = max(Otsu, 0.5×p95)（制版 50% 外形语义；Otsu 单独用会被"浅裙边大峰 vs 线深小峰"的不平衡拖低），外形候选 = 半深度区域中与管线响应区连通的连通域（原图定外形、检测定取舍：深度足够但无响应的暗斑被剔除）；面积骤降 40% 以上回退滞回区域（单峰细线保护）
- **测地重建墨度**（`_geodesic_reconstruct`）：DoG 对宽线中心零响应，response 模式以边缘响应为种子在外形内做形态学测地重建（by dilation）填满线心；小碎片零区（AA 边/响应缺口）单次 5×5 传播填平、大零区逐连通域 bbox 局部迭代（全图迭代版曾致 2048 图 5.4s，bbox 版恢复至 2.9s）
- **结构**：渲染双路线分流（pipeline 按 `render_mode` 分支，layered/color 提前返回消除无用的 mask 计算）；`_render_grayscale` 尾部抽公共 `_to_white_black_u8`；版本三处统一 1.10.0
- **测试**：新增 tests/test_shape_route.py 12 项（默认 splat 逐位零回归、外形 IoU、变宽线跟随、圆位置、三墨度模式、binary、语义选择器、非法参数拒绝、空白图）；压测 54 → 60 项（shape×response/source/flat/binary/closing0 + 4096² shape）；perf_v13 精度逐位一致（0.0615/0.2256/0.2605px）

### v1.9.2
- **性能修复 14.6x**：4096×3072 密集线稿（260 线 + 60 圆，骨架端点 68k）全管线 68.8s → **4.7s**；此前 v1.9.1 的 3s 数据基于端点稀疏图，端点密集图上 A2 断线桥接退化到 65s
- **修复桥接匹配 O(N²) 性能缺陷**（`connectivity.py`）：Python 逐端点暴力搜索 + 标量 `np.hypot`（68k 端点 46 亿次标量运算，65s）改为空间网格（边长 max_gap）分桶 + 向量化距离 + 按 (端点序, 距离) 排序贪心扫描（0.36s，165x），贪心语义与原版逐端点最近邻完全等价
- **修复 Bresenham 斜线正确性缺陷**：`_bresenham_line` 的斜率累加 `y += slope` 误写在列表推导之外且仅执行一次，桥接线全部退化为水平/垂直线而非端点连线
- **删除无效 ETF 计算**：`bridge_broken_lines` 简化版实际不使用流场（仅距离约束），pipeline 中为传 tx/ty 做的 4096 全分辨率 `compute_etf`（~1.4s/次）为纯浪费，连带删除 `_build_mask` 的 tx/ty 死参数
- **修复打包配置真实缺陷**：`pyproject.toml` packages 列表缺 `lineart.io` / `lineart.gui`（v1.9.0 新增子包，此前 pip 安装会丢失 SVG/GUI/视频功能）；新增 `accel` 可选依赖组（numba）
- **结构清理与版本统一**：版本号三处不一致（`__init__.py` 1.8.2 / `pyproject.toml` 1.8.1 / git v1.9.1）统一为 1.9.2；删除纯重复代码 `denoise.guided_filter_boxfilter`（v1.9.1 回退后与主实现逐行相同）；删除 2 个过时基准脚本（`bench_integral.py` 对比对象为已否决方案、`bench_new_scenes.py` 引用已合并符号处于 ImportError 状态）；.gitignore 补 `__pycache__/`（32 个 .pyc 此前被误跟踪）
- **评估后不采用**：`bilinear_sample` 改 `cv2.remap`（可省 ~0.8s），因 remap 全 float32 而手写版 float64 权重，会在亚像素精化路径引入数值差异——按"精度 > 性能"优先级放弃
- 精度回归：perf_v13 六节逐位一致（细线 0.0615px / 模糊边 0.2256px / 圆半径 0.2605px）

### v1.9.1
- **性能优化 11.5x**：4096×3072 全管线 35052ms → 3038ms，精度逐位一致（perf_v13 细线 0.0615px / 模糊边 0.2256px 零回归）
- **denoise 回退积分图**：自研 `_box_integral_optimized`（4096 图 676ms）比 `cv2.boxFilter`（47ms）慢 14 倍，删除积分图实现保留 boxFilter
- **ETF 参数收敛**：默认 kernel 5→3、iters 4→1，调用处按相干度自适应 `iters=1+round(2*coh)`；大图（>512²）用 `cv2.filter2D` 单位脉冲核替代 `_shift` 迭代（26ms vs 137ms/次）
- **ETF 延迟计算**：管线仅在 `merge_edges and bridge_gap>1` 时才计算 ETF（原无条件调用是最大热点）

### v1.9.0
- **A 精度增强**：A1 多尺度语义融合（`flow.multiscale`，1/1÷0.5÷0.25 三尺度加权 0.5/0.3/0.2，弱对比度线检出提升）；A2 断线桥接（`localization.connectivity.bridge_broken_lines`，骨架化后端点近邻连接，`bridge_gap=8`，closing_size=1 防细线粘连，无 scipy 依赖）；A3 自适应线宽（`render._fwhm_widths` 按梯度硬/软边取 0.4/0.6 阈值，`adaptive_width=True`）
- **B 性能**：numba JIT `_shift`（未安装自动回退）；`bilinear_splat_batch`（np.bincount 批量 splat）；（B2 积分图方案后被 v1.9.1 实测否决回退）
- **C 功能**：SVG 矢量输出（`io/svg_export.py`，道格拉斯-普克简化）；批量 API（`to_lineart_batch`，ThreadPoolExecutor）；视频/GIF 逐帧处理（`io/video.py`，ETF 帧间融合 α=0.3 防闪烁）；tkinter 实时预览 GUI（`gui/preview.py`，12 参数滑动条）
- **D 场景（未落地）**：comic/handwriting/architectural 检测在开发验证中因与既有场景的误判链（sketch_photo 曾被连环误判）于提交前回退，**未合入 v1.9.0 提交**（历史核查：v1.9.0~v1.10.0 各版本 scenes.py 均无此三场景），场景检测实际保持 5 场景；其遗留基准脚本 bench_new_scenes.py（引用不存在的检测函数，ImportError 状态）已在 v1.9.2 删除
- 修复实施过程中的循环导入（io 包化）、OpenCV 5.0 `HoughLinesP` 参数名变化、断线桥接破坏 sketch_photo 细线等问题

### v1.8.1
- **删除死文件 `lineart/precise.py`**（295 行）：v1.1.0 已合并进 subpixel 的遗留，v1.4.0/v1.6.0 两次记录删除，均被环境重置的旧快照复活；全库 grep 确认零引用
- **移除 11 个旧平铺兼容 shim**（adaptive/background/edges/flow/guided/illumination/neural/shock/subpixel/text/xdog，v1.4.0 为平滑迁移保留）：项目自研无外部用户，包结构自 v1.4.0 起已稳定 5 个版本；现存引用同步迁移至子包路径（test_scenes.py → `semantics.adaptive`、perf_v13.py → `preprocess.shock`、`__main__.py` → `semantics.neural`——后者本是重组时漏改的旧路径）
- **删除 3 个已损坏的一次性等价验证脚本**（benchmarks/perf_equiv*.py）：其引用的私有符号（`_otsu_hi`/`_fast_median`/`_corner_stats` 等）在 shim 中并不导出，v1.4.0 重组后即处于 ImportError 状态，等价性结论已记录在案
- **清理生成物**：__pycache__×6、.pytest_cache、demo/output/（860K 演示输出，均在 .gitignore 内）
- 验证：26 项单测（含 15 子测试）+ perf_v13 六节精度逐位一致（0.0615/0.2256px）+ 54 项压力测试全过 + CLI 冒烟（`python -m lineart`）正常

### v1.8.0
- **背景估计有效 σ 修正**（`preprocess/illumination.py`）：旧实现 σ 乘 √2（"两次高斯合成一次"的历史遗留，但函数只 blur 一次），等效平滑尺度偏大 √2 倍——大图背景场与全分辨率大核高斯基准（σ=max_dim×0.035）的偏差 max 0.032（blueprint 放大 2048 实测），阴影跟随失真。去掉后偏差降至 0.002；同时引入绝对下限 `min_sigma=2.0`：小图（如 192px 输入 → 1/8 尺度仅 24px、比例 σ 不足 1.0）上圆等大结构会泄漏进背景场、除法校正抹平内容对比度，实测 σ≥1.5 后泄漏饱和消失，取 2.0 留裕量（只影响 ≤457px 输入，大图比例 σ 远大于此）。预滤波方案（dilate/close 抹暗线）实测全面失败：把真实线条也抬进背景，大图偏差反而恶化至 0.155
- **合并去重组内位置平均**（`localization/render.py`）：同一半像素网格的多对合并中点，位置由"组首"（依赖输入顺序、丢弃组内亚像素信息）改为组内平均（顺序无关）；法向仍取组首（避免组内法向冲突相消）。浓度/线宽维持组内最大
- **语义层大图降采样 0.35→0.25**（`semantics/flow.py`，m>2560 档）：4096 图全管线约 24% 提速（2349→1779ms 同进程 A/B）。质量证据（墨定义 = 输出 <128，注意 `>=128` 是白背景占比 97%+，方向搞反会得出 IoU≈1 的假象）：真实样本放大 4096 后 3px 容差 IoU ≥0.997（4/6 样本）；cad_thick_lines 的 0.9435 差异经连通域 + 原图灰度验证为**正向修复**——0.35 漏检了灰度 80/背景 240 的 3px 网格线段（121 个 60~267px 水平线段），0.25 全部检出，且 0.25 丢失方向仅 148px 碎片（无结构丢失）
- **速率优化三项**（`common.py` / `semantics/flow.py`）：bilinear_splat 四角索引/权重拼接后单次 `np.add.at`（组件级 3.5x，167.5→48.2ms，数值等价 max|Δ|=1.2e-07）；`_shift` 改 `empty_like` + 边缘 4 条带清零（省全图 memset，1.7x）；`_tanh` 原生 np.tanh 替换 LUT（NumPy 2.x 对 float32 有 SIMD，更快且消除 4.9e-3 量化误差）
- **端到端精度（perf_v13 同环境 A/B，v1.7.1 基线 → v1.8.0）**：细线亚像素 0.0840→**0.0615px**（-27%）、模糊边端到端 0.2861→**0.2256px**（-21%）、圆半径误差 0.3558→**0.2605px**（-27%）、consensus 圆 1.587→**1.205px**（-24%），六节全线改善零回归；4096 大图与基线结构一致性：3px 容差 IoU 0.998~1.000（3/4 样本，差异为 σ 修正引起的 ~1px 边缘位移，方向已证更准）
- **项目结构**：10 个实验/基准脚本收纳至 `benchmarks/`（work_scale 决策、背景上采样、预滤波对比、micro-bench、cProfile、终验对比等，根目录只留 perf_v13.py 与 stress_test.py）；新增 `pyproject.toml`（包元数据 + 依赖声明 + pytest 配置）
- 回归：26 项单测（含 15 子测试）+ perf_v13 六节 + 54 项压力测试全过

### v1.7.0
- **粗线双边缘合并**（`localization/merge.py` 新增，`render.py` 集成）：边缘检测的本质（梯度脊线）对线宽 ≥2px 的线给出两条脊线，直接渲染把一根线画成两根伴随线（3px 硬边线实测：真线丢失 99%、虚假墨 1460px）。合并管线：脊线点沿梯度负方向搜索对面脊线（法向点积 < -0.5 反向验证 + 中点暗谷判据区分粗线两边缘与近距平行细线）→ 亚像素中点即线心 → 半像素网格去重 → FWHM 线宽展开渲染。合并后虚假墨 1460→19px（-99%）、丢线 99%→3%、单线位置与真实墨逐一吻合，且对高斯噪声完全鲁棒（干净/噪声指标一致）
- **FWHM 线宽测量**（`render.py`）：合并线的渲染宽度不从脊线间距推导（硬边线间距≈宽、平滑细线间距≈宽+1，无一致映射），而是从亮度场直接测半高宽——线心浓度 c、远端背景 bg、阈值 c+0.5(bg-c)、两侧档间线性插值，物理正确且对软硬线统一
- **farid 梯度符号修正**（`gradient.py`）：`_FARID_DERIV` 原始核在 cv2 correlation 语义下输出 -∇I（与 Sobel 的 +∇I 相反），取反后恢复论文语义（+∇I）；此前未暴露是因为多数模块只用幅值/对称采样
- **`_render_binary` 黑白反转修复**（`pipeline.py`）：二值输出 `>=128 → 0(黑)` 写反，应为 `>=128 → 255(白)`；此前 masked 由 antialiased 默认路径掩盖
- 新增参数：`merge_edges`（默认 True）、`merge_gap`（默认 6）；pixel 场景显式关闭合并（最近邻预放大管线保持原行为）
- 新增 `tests/test_merge.py`（4 项 + 2 子测试）：结构保真回归（含 v1.6.0 缺陷基线断言）

### v1.6.0
- **渲染边界伪影修复**（`common.py`，`bilinear_sample` / `bilinear_splat`）：越界权重/采样不再累加到边界像素（旧版 `np.clip` 把边缘线拉向画布中心），改为丢弃越界权重 + 返回 0。亚像素精化在图像边界的精度提升，边缘线位置不再被边界像素"吸附"
- **边界精化鲁棒性**：5 点高斯拟合在边界处自动退化为 3 点（越界采样返回 0，拟合系数 a 自动 <0 → delta=0 安全回退），避免边界处的异常偏移
- **删除死文件 `precise.py`**（第 9 轮后复活，无任何引用）

### v1.5.0
- **压力测试**（`stress_test.py`，54 项 0 失败）：尺寸（1×1~8192²）/ 内容（纯色、噪声、渐变、棋盘）/ dtype（uint8/float32/float64/2D 灰度/单通道/RGBA）/ 参数极端值 / 语义×场景×输出组合 / 内存稳定性（连续 30 次 RSS 0 增长）/ 超大图性能上限 六维度覆盖
- **输入标准化 `coerce_bgr`**（io.py）：统一把任意输入宽容转换为 uint8 BGR——float64/float32 按值域归一化、二维灰度与单通道 3D 转 BGR、RGBA alpha 合成白底、None/1D/空图明确 `ValueError`（此前这些输入分别导致 cv2 报错、IndexError、ZeroDivisionError）；`load_image` 与 `to_lineart` 入口共同复用
- **底色检测除零保护**：空图直方图 `hist.sum()=0` 导致 divide warning + ZeroDivisionError，加 `max(sum, 1e-12)`
- **shock 精度回归修复**：`deriv_ksize` 5→3（第 8 轮性能优化时改）使法向二阶导零交叉偏离真实拐点，端到端精度 0.2861→0.3066px；改回 5x5 恢复 0.2861px（性能由 iter=2 + 结构张量复用弥补，4096 图 shock 单发 1209ms）

### v1.4.0
- **分层结构重组**（17 个平铺模块 → 按管线阶段分层的 3 个子包，参照 scikit-image 的分层组织模式）：
  - `preprocess/`（背景/光照/去噪/文字/冲击锐化）→ `semantics/`（FDoG/XDoG/Sauvola/神经先验）→ `localization/`（梯度/NMS/精化/渲染），目录结构即管线阶段顺序，与「语义-定位解耦」架构一一对应
  - 394 行的 `subpixel.py` 按职责拆为 `localization/{gradient, refine, render}.py`（梯度算子 / 亚像素精化 / 渲染）
  - `LineArtParams` 从 pipeline.py 拆出为 `params.py`（参数契约与编排解耦，CLI/API/测试共享唯一来源）
  - `guided.py` 更名 `preprocess/denoise.py`（名称对齐职责）
  - **100% 向后兼容**：11 个旧平铺路径（subpixel/xdog/flow/adaptive/edges/...）保留为 re-export shim，tests/demo/perf 脚本零修改通过（`from lineart.shock import coherence_shock` 等旧用法不变）
  - 删除死文件 `precise.py`（v1.1.0 已合并进 subpixel 的遗留，无任何引用）
  - 验证：22/22 测试通过；perf_v13 六节精度指标与重组前逐一相同；demo 全样本场景检测一致；旧路径 16 项 + 新路径 4 项导入冒烟 + shim 同一性（is）校验通过

### v1.3.1
- **相干增强冲击锐化**（`shock.py`，Weickert DAGM 2003）：模糊输入的过渡带收窄成准阶跃（实测 4px → 0px），语义检测更稳、输出线更黑
  - **值推格式**（非形态学）：凸区 +dt / 凹区 -dt 沿结构张量法向对称推进，拐点灰度不变（经典 dilate/erode 实现每步强制位移 1px，实测漂移 257~300 mpx）
  - **检测/定位数据流分离**：锐化图只用于语义提取与验证强度；亚像素拟合场始终用原始图（冲击会把中间灰度推向 0/1，位置量化到整数网格）——端到端精度与 sharpen=0 完全一致（0.2861px，模糊边基线）
  - **陡阈值门控**（|∇I|/peak < 0.15 完全不推）：预处理残余噪声免疫（宽松门控曾把噪声二值化放大成伪线）
  - 性能：4096×3072 + sharpen=0.6 全管线 3217ms（迭代 4→2、导数核 5→3、结构张量跨迭代复用，3.9x 提速）
- **SSR 超采样渲染**（`render_ss`，默认 2）：粗线（thickness>1）在 2x 画布 splat 后 INTER_AREA 下采样，线宽与边缘抗锯齿显著更准（thickness=3 质心误差 0.066→0.015px）；细线自动回退 1x（单点 splat 在 SSR 网格上被量化到半像素网格，实测 0.063→0.169px）
- **亚像素偏移中值共识**（`consensus`，默认开，SER 思想 arXiv:2502.16502）：3x3 邻域有线点 delta 中值作共识，偏离 >0.2px 的孤立点替换为共识（离群 0.5px → 0.1px 完全拉回，连续渐变零影响）
- **场景检测修正**：双色调半平面/大色块图（每条扫描线跳变位置几乎相同）是 1D 边缘结构而非像素艺术 2D 块状图案，新增单一长直边排除判据；pixel 预放大管线自动禁用 SSR（软墨破坏 block-max 硬边缩回）
- **管线修正**：sharpen 移到文字去除之后（原顺序下 inpaint 重算灰度会静默丢弃锐化）；`subpixel.py` 清理 return 后死代码

### v1.3.0
- **全链路性能优化**（3.0x，数值等价性逐项验证，最终输出 0 差异）：

| 场景 | v1.2.0 | v1.3.0 | 加速 |
|---|---|---|---|
| 1600×1200 auto（sketch/adaptive） | 802 ms | **165 ms** | 4.9x |
| 1600×1200 off（fdog） | ~1350 ms | **412 ms** | 3.3x |
| 4096×3072 auto | 4453 ms | **1492 ms** | 3.0x |

- **语义/统计层低尺度化**：Sauvola 阈值场与软过渡 ink 在 0.75 尺度计算（boxFilter 替代 float64 积分图，351ms → 65ms）；背景估计 1/8 尺度（两次高斯合成一次 σ√2）；场景/底色检测统一 ≤320 降采样统计
- **导向滤波自引导快速路径**：corr_ip ≡ corr_ii，6 次 box + 2 次全图乘法 → 3 次 + 1 次
- **向量化收尾**：滞回 BFS → connectedComponents；NMS 稀疏候选 + tan 替代 arctan2；tanh/gamma 查表；高斯拟合克莱姆法则；cv2.remap 双线性采样；连通域清理标签 LUT；findNonZero 替代 np.nonzero
- **大图 FDoG 分档修正**：work_scale 增加 0.35 档（>2560px），修正嵌套三元条件写反导致的分档不可达 bug

### v1.2.0
- **场景自适应**：6 类场景自动检测（拍照手绘/扫描文档/蓝图/像素艺术/数字线稿/照片），参数与预处理自动适配
- **Sauvola 自适应语义层**（`adaptive.py`）：积分图 O(1)/像素，阴影下线响应强度一致
- **蓝图支持**（`background.py`）：蓝底白线 → 白底黑线（V×(1-S) 提取 + 背景平坦化）
- **像素艺术支持**：最近邻 4x 预放大产生可精化过渡带，输出缩回原尺寸（API 契约不变）
- **清理遗留**：删除重构残留的 `precise.py`

### v1.1.0
- **创建 `common.py`**：集中公共工具函数，消除 4 个文件中的重复代码
- **合并 `precise.py` → `subpixel.py`**：统一梯度算子与精化算法接口
- **修复向量化高斯拟合 bug**：最小二乘公式从 2x2 修正为 3x3 系统
- **向量化 gaussian 精化**：100x 加速
- **总代码量**：1637 行（比 v1.0 减少约 10%）

### v1.0.0
- 初始版本：语义-定位解耦架构
- 支持 FDoG/XDoG/神经先验三种语义
- CAD 增强：底色反转、多色线、网格去除、文字去除、中心线提取
- 精度：平均误差 0.040px，散布 0.101px

## 已知边界

- 结构对比度低于背景纹理（SNR<1.3）时召回率下降（信息论极限附近）
- FDoG 与 XDoG 在本架构下输出高度一致（门限机制承担了主要纹理抑制），FDoG 提供低对比度场景的语义冗余与流线连贯参数空间
- 风格化程度（笔触感）不如 GAN 类方案——本项目目标是几何精度而非艺术风格
