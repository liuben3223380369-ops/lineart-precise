# lineart-precise Codebase 文档

> 📁 本文档为项目总览：技术栈、目录结构、关键模块与开发命令。当前版本 v1.10.0（pyproject.toml / git HEAD 149be70）。主源码树展开未超过 40 行，无需拆分子文档。

## 1. 技术栈

- **Python >= 3.9**（来自 pyproject.toml `requires-python`；当前工作区实测 Python 3.12.5，解释器路径 `/opt/cloud/devTools/python/3.12.5/bin/python3`）
- **包管理器 / 构建后端**：pip + setuptools >= 61（pyproject.toml `[build-system]`，build-backend = setuptools.build_meta，纯 Python 包无编译步骤）
- **核心依赖**：opencv-python-headless >= 4.5 + numpy >= 1.20（纯传统 CV 算法实现，无需深度学习模型权重，离线可用）
- **可选加速**：numba >= 0.58（pyproject.toml `[project.optional-dependencies] accel`；JIT 加速 ETF 与 `_shift`，未安装时自动回退纯 NumPy）
- **测试**：pytest >= 7（pyproject.toml `dev` 可选依赖组；`[tool.pytest.ini_options]` testpaths = ["tests"]）
- **GUI**：tkinter（Python 标准库，实时预览调参）

## 2. 项目目录结构

```
lineart-precise/
├── lineart/            # 主包：图片 → 白底黑线高精度转换管线，「语义-定位解耦」架构（优先级：精度 > 清晰度 > 性能）；公共 API（to_lineart / to_lineart_batch / LineArtParams）与 CLI 入口在包顶层
│   ├── preprocess/     # 预处理层：任意输入 → 干净灰度场——底色检测与黑/白/灰/蓝图自适应反转、1/8 尺度光照归一化、导向滤波保边去噪、MSER+inpainting 文字区域去除、相干冲击锐化（检测/定位数据流分离）
│   ├── semantics/      # 语义层：灰度 → 线响应图，判定「哪里有线」（可插拔）——ETF 流场 + FDoG 流线语义、多尺度软阈值 XDoG、Sauvola 自适应阈值（拍照手绘/扫描文档）、外部神经先验接口（Anime2Sketch / DexiNed 等）
│   ├── localization/   # 定位与渲染层：线响应 → 亚像素线稿，保证「线在哪里」——Farid/Sobel 梯度算子、梯度验证/NMS/滞回/骨架/连通域清理、断线桥接、亚像素精化（抛物线/高斯/Zernike + 中值共识）、粗线双边缘合并、抗锯齿 splat 渲染 + shape 外形保真渲染双路线
│   ├── io/             # 读写与导出子包：图像读写（中文路径/RGBA 合成白底/任意 dtype 宽容转换）、SVG 矢量输出（道格拉斯-普克折线简化）、视频/GIF 逐帧处理（ETF 帧间一致性防闪烁）
│   └── gui/            # tkinter 实时预览 GUI：12 参数滑动条实时调参
├── tests/              # 单元测试（pytest，38 项 + 15 子测试）：亚像素精度与鲁棒性、场景检测与转换质量、粗线合并结构保真回归、shape 外形保真渲染路线回归
├── benchmarks/         # 实验/基准脚本（19 个独立脚本）：work_scale 决策、背景估计上采样/预滤波对比、micro-bench、cProfile 热点、ETF 提速、批量/多尺度/SVG/自适应线宽/shape 路线量化验证等；含 perf_v13.py（六节精度基准）与 stress_test.py（60 项压测，模块级自执行、放 tests/ 会被 pytest 误收集执行）
├── demo/               # 合成场景与演示：14 个合成样本图（拍照手绘/扫描文档/蓝图/像素艺术/CAD/照片等）、样本生成与全场景演示脚本；output/ 为演示输出产物（生成物）
├── pyproject.toml      # 包元数据 + 依赖声明（core/accel/dev 三组）+ 6 个打包子包声明 + pytest 配置
├── requirements.txt    # 运行依赖清单（opencv-python-headless / numpy / numba 可选）
└── README.md           # 项目文档：全网方案调研、架构说明、参数表、精度实测、v1.0.0 起完整重构历史
```

### 关键模块说明

- **lineart/pipeline.py** — 主管线编排：预处理 → 语义 → 定位 → 渲染全流程调度，含场景自适应分支、渲染双路线（splat 中心线重绘 / shape 外形保真）分流、灰度/二值输出与批量 API
- **lineart/params.py** — 参数契约：LineArtParams 为全部管线参数的唯一来源，CLI（__main__.py）/ API / 测试三方共享
- **lineart/scenes.py** — 场景自动检测与参数适配：5 种检测场景（sketch 拍照手绘与扫描文档 / blueprint 蓝图 / pixel 像素艺术 / lineart 数字线稿与 CAD / photo 照片）。注：comic/handwriting/architectural 曾在 v1.9.0 开发中实验，因与既有场景误判链在提交前回退，从未合入任何版本（见 README.md > 重构历史 v1.9.0）
- **lineart/semantics/** — 语义层（4 模块）：flow（ETF+FDoG 流线语义，多尺度融合）、xdog（多尺度软阈值）、adaptive（Sauvola 积分图二值化）、neural（外部神经先验接口）
- **lineart/localization/** — 定位与渲染层（7 模块）：gradient（梯度算子）、edges（NMS/滞回/骨架/清理）、connectivity（断线桥接）、refine（亚像素精化 + 中值共识）、merge（粗线双边缘合并）、render（splat 抗锯齿渲染 + SSR 超采样 + FWHM 自适应线宽）、render_shape（v1.10.0 新增外形保真渲染路线：原图墨度半深度外形为主、检测响应为辅）
- **lineart/preprocess/** — 预处理层（5 模块）：background（底色检测与反转）、illumination（光照归一化）、denoise（导向滤波保边去噪）、text（文字检测去除）、shock（相干冲击锐化）
- **lineart/io/** — 读写与导出：图像读写（中文路径/Alpha 合成白底）、svg_export（SVG 矢量输出）、video（视频/GIF 逐帧处理）
- **lineart/gui/preview.py** — tkinter 实时预览（12 参数滑动条）
- **lineart/common.py** — 公共工具（双线性采样/splat、形态学、连通域）；**lineart/color.py** — 多色线分离与颜色保留渲染
- **根目录 perf_v13.py** — 六节端到端精度基准（顶层自执行脚本）；**根目录 stress_test.py** — 60 项压力测试（顶层自执行脚本：尺寸 1x1~8192 平方 / dtype / 参数极端值 / 语义×场景×输出组合 / 内存稳定性 / 4096 平方超大图）

## 3. 开发命令（核心）

### 构建 / 安装命令

```bash
# 安装依赖（来源：README.md > 安装与使用）
pip install -r requirements.txt
```

### 测试命令

```bash
# 单元测试：38 项 + 15 子测试（来源：README.md > 测试与演示 `python -m pytest tests/ -q`；已用工作区 Python 3.12.5 实测验证：38 passed, 15 subtests passed）
/opt/cloud/devTools/python/3.12.5/bin/python3 -m pytest tests/ -q

# 演示脚本：全部合成场景 + 三模式对比 + 场景自适应演示（来源：README.md > 测试与演示）
python demo/run_demo.py

# 端到端精度基准：六节（细线亚像素 / 模糊边 / 圆半径等）（benchmarks/perf_v13.py，路径已相对化，任意 cwd 可跑）
python benchmarks/perf_v13.py

# 压力测试：60 项——尺寸 / 内容 / dtype / 参数极端值 / 组合 / 内存稳定性 / 超大图（benchmarks/stress_test.py，模块级自执行，勿经 pytest 运行）
python benchmarks/stress_test.py
```

### 运行 / 启动命令

```bash
# 单张转换（默认 FDoG 流线语义）（来源：README.md > 安装与使用）
python -m lineart photo.jpg -o lineart.png

# 目录批量（多线程）+ 纯黑白 + 加粗（来源：README.md > 安装与使用）
python -m lineart ./photos -o ./results --mode binary --thickness 2 --batch --workers 4

# 外形保真渲染路线（宽线/笔触/制版保真）（来源：README.md > 安装与使用）
python -m lineart in.jpg -o out.png --render-mode shape --shape-ink source

# SVG 矢量输出（来源：README.md > 安装与使用）
python -m lineart in.jpg -o out.svg --output-svg

# 实时预览 GUI（来源：README.md > 安装与使用）
python -m lineart --gui
```

> 注：README 中命令统一写作 `python`；本工作区解释器为 `/opt/cloud/devTools/python/3.12.5/bin/python3`（Python 3.12.5），实际运行时可用该路径替换 `python`。CLI 全部参数说明见 README.md > 参数表。

## 4. 开发环境（简要）

### 前置要求

- Python >= 3.9（当前工作区 3.12.5：`/opt/cloud/devTools/python/3.12.5/bin/python3`）
- pip + setuptools >= 61
- 无需数据库 / GPU / 网络（纯 OpenCV + NumPy 传统算法，离线可用；numba 为可选加速，未安装自动回退）

### 配置说明

- `pyproject.toml` — 包元数据、打包子包声明（lineart 及 preprocess / semantics / localization / io / gui 五个子包）、core / accel / dev 依赖组、pytest testpaths
- `requirements.txt` — 运行依赖清单（numba 以注释标明可选回退语义）
- 无 `.env` / 数据库 / 外部服务配置文件
