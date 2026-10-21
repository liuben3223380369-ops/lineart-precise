# -*- coding: utf-8 -*-
"""文件名: preview.py
描述: 线稿转换桌面 GUI（v1.27.0 完全重构）

架构：
- 顶栏工具栏：打开图片 | 转换（后台线程）| 导出 PNG | 运行自检 + 模型状态指示
- 左侧参数面板（Notebook 两页）：
  * 语义页：语义选择器（fdog/xdog/adaptive + 神经三档 neural/dexined/ensemble——
    v1.24.0 起 EXE 内置双 fp32 权重，ensemble 开箱即用）、场景路由、
    输出模式、DexiNed 推理倍率
  * 参数页：传统链可调参数滑动条（神经链下自动禁用——对输出零影响）
- 中心预览画布：原图/线稿切换、滚轮缩放、拖动平移、适应窗口/1:1
- 底部状态栏：模型档位 | 图像尺寸 | 耗时

v1.27.0 重构动机（用户实测反馈）：
- 旧版无「转换」按钮（滑动条自动重跑，大图/神经推理卡死 UI）
- 无「自检」按钮（selfcheck 只有 CLI 入口）
- 无模型选择（EXE 明明内置双权重，神经语义在 GUI 无入口）
- 转换移到后台线程 + 显式按钮，参数变动只标记「需重新转换」
"""
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

import cv2
import numpy as np

from ..params import LineArtParams
from ..pipeline import to_lineart
from ..io import load_image, save_image
from ..io.video import process_video, process_gif, is_supported_video

SEMANTIC_CHOICES = [
    ("fdog", "fdog 流线连贯（默认）"),
    ("xdog", "xdog 经典多尺度"),
    ("adaptive", "adaptive 自适应阈值"),
    ("neural", "neural 线稿模型（内置 17MB）"),
    ("dexined", "dexined 边缘模型（内置 45MB）"),
    ("ensemble", "ensemble 三语义融合（最优，推荐）"),
]
SCENE_CHOICES = [
    ("auto", "auto 自动检测"), ("off", "off 关闭"),
    ("pixel", "pixel 像素图"), ("blueprint", "blueprint 蓝图"),
    ("sketch", "sketch 手绘"), ("lineart", "lineart 线稿"),
    ("photo", "photo 照片"),
]
OUTPUT_CHOICES = [
    ("grayscale", "grayscale 白底黑线"),
    ("binary", "binary 纯黑白"),
    ("color", "color 彩色线（CAD）"),
    ("layered", "layered 分层输出"),
]
DEX_SCALE_CHOICES = [
    ("auto", "auto 场景路由"), ("1", "1 倍（480×640）"),
    ("2", "2 倍（高分辨率，更清晰）"),
]
NEURAL_SEMANTICS = ("neural", "dexined", "ensemble")


def _model_status_text() -> Tuple[str, str]:
    """检测内置/已配置神经权重档位 → (状态文案, 语义推荐)。

    EXE v1.24.0 起双 fp32 权重随包（_MEIPASS/models/），源码运行
    命中仓库根 models/——ensemble 理论上开箱即用；此处如实探测
    并把最高可用档作为推荐默认。"""
    try:
        from ..semantics.neural import (available, dexined_available,
                                        ensemble_available)
        if ensemble_available():
            return ("模型：双权重就绪（线稿 17MB + DexiNed 45MB）"
                    "→ ensemble 可用", "ensemble")
        if dexined_available() and available():
            return "模型：双权重存在但 ensemble 不可用", "dexined"
        if available():
            return "模型：线稿模型就绪（DexiNed 缺失）", "neural"
        if dexined_available():
            return "模型：DexiNed 就绪（线稿模型缺失）", "dexined"
        return "模型：无神经权重（传统语义可用）", "fdog"
    except Exception:
        return "模型：检测失败（传统语义可用）", "fdog"


class LineArtPreviewGUI:

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LineArt 线稿转换器 — 高精度 白底黑线")
        root.geometry("1180x760")

        # 状态
        self.original_img: Optional[np.ndarray] = None
        self.processed_img: Optional[np.ndarray] = None
        self.input_path: Optional[str] = None
        self.neural_map: Optional[np.ndarray] = None
        self.params = LineArtParams()
        self.dirty = False          # 参数已改待重转换
        self.converting = False     # 后台转换中
        self._result_q: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.zoom_scale = 1.0
        self._fit_scale = 1.0
        self._tk_photo = None
        self._sliders = []

        # 模型探测 → 推荐默认语义（EXE 双权重 → ensemble）
        model_text, recommended = _model_status_text()
        self.params.semantic = recommended

        self._create_widgets(model_text)
        self._bind_events()
        self._poll_results()
        self.status_var.set(f"就绪 — {model_text}")

    # ------------------------------------------------------------ UI 构建

    def _create_widgets(self, model_text: str):
        """顶栏工具栏 + 左参数面板 + 中心预览 + 底部状态栏。"""
        # ── 顶栏工具栏（v1.27.0：转换/自检/模型状态首次进 GUI）──
        toolbar = ttk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(6, 2))
        self.btn_open = ttk.Button(toolbar, text="打开图片",
                                   command=self.open_image)
        self.btn_open.pack(side=tk.LEFT, padx=2)
        self.btn_convert = ttk.Button(toolbar, text="▶ 转换",
                                      command=self.convert_now)
        self.btn_convert.pack(side=tk.LEFT, padx=2)
        self.btn_export = ttk.Button(toolbar, text="导出 PNG",
                                     command=self.export_image)
        self.btn_export.pack(side=tk.LEFT, padx=2)
        self.btn_selfcheck = ttk.Button(toolbar, text="运行自检",
                                        command=self.run_selfcheck)
        self.btn_selfcheck.pack(side=tk.LEFT, padx=2)
        self.model_label = ttk.Label(toolbar, text=model_text,
                                     foreground="#0a7")
        self.model_label.pack(side=tk.RIGHT, padx=6)

        # ── 主体：左参数面板 + 右预览 ──
        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # 左侧参数面板（Notebook：语义 / 传统参数）
        panel = ttk.Frame(body, width=280)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        panel.pack_propagate(False)
        notebook = ttk.Notebook(panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        self._create_semantic_page(notebook)
        self._create_param_page(notebook)

        # 右侧预览区
        preview_frame = ttk.Frame(body)
        preview_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 预览模式切换
        mode_bar = ttk.Frame(preview_frame)
        mode_bar.pack(fill=tk.X, pady=(0, 2))
        self.preview_mode = tk.StringVar(value="processed")
        ttk.Radiobutton(mode_bar, text="线稿结果", value="processed",
                        variable=self.preview_mode,
                        command=self.update_preview).pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(mode_bar, text="原图", value="original",
                        variable=self.preview_mode,
                        command=self.update_preview).pack(side=tk.LEFT, padx=2)

        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1,
                                highlightbackground="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        view_bar = ttk.Frame(preview_frame)
        view_bar.pack(fill=tk.X, pady=(2, 0))
        ttk.Button(view_bar, text="适应窗口", width=10,
                   command=self.zoom_fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(view_bar, text="1:1 原始像素", width=12,
                   command=self.zoom_actual).pack(side=tk.LEFT, padx=2)
        ttk.Label(view_bar, text="滚轮缩放 · 按住左键拖动平移",
                  foreground="#666").pack(side=tk.LEFT, padx=8)

        # ── 底部状态栏 ──
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var,
                               relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _create_semantic_page(self, notebook: ttk.Notebook):
        """语义页：语义/场景/输出/推理倍率（神经语义首进 GUI）。"""
        page = ttk.Frame(notebook, padding=8)
        notebook.add(page, text="语义")

        ttk.Label(page, text="语义选择（哪里有线）",
                  font=("", 10, "bold")).pack(anchor=tk.W)
        self.semantic_var = tk.StringVar(value=self.params.semantic)
        for value, text in SEMANTIC_CHOICES:
            ttk.Radiobutton(page, text=text, value=value,
                            variable=self.semantic_var,
                            command=self.on_param_change).pack(
                anchor=tk.W, pady=1)

        ttk.Separator(page).pack(fill=tk.X, pady=6)

        def _combo(label, choices, var, width=18):
            row = ttk.Frame(page)
            row.pack(fill=tk.X, pady=2)
            ttk.Label(row, text=label, width=10).pack(side=tk.LEFT)
            cb = ttk.Combobox(row, textvariable=var, state="readonly",
                              width=width, values=[c[1] for c in choices])
            cb.pack(side=tk.LEFT)
            cb.bind("<<ComboboxSelected>>",
                    lambda _e: self.on_param_change())

        self.scene_var = tk.StringVar(
            value=next(t for v, t in SCENE_CHOICES if v == "auto"))
        _combo("场景", SCENE_CHOICES, self.scene_var)

        self.output_var = tk.StringVar(
            value=next(t for v, t in OUTPUT_CHOICES if v == "grayscale"))
        _combo("输出模式", OUTPUT_CHOICES, self.output_var)

        self.dex_scale_var = tk.StringVar(
            value=next(t for v, t in DEX_SCALE_CHOICES if v == "auto"))
        _combo("推理倍率", DEX_SCALE_CHOICES, self.dex_scale_var)

        ttk.Separator(page).pack(fill=tk.X, pady=6)
        ttk.Label(page, foreground="#888", wraplength=240, justify=tk.LEFT,
                  text="神经链（neural/dexined/ensemble）内置权重开箱即"
                       "用；ensemble 三语义融合细节召回最高。神经链下"
                       "「参数」页滑动条无效（位置由模型响应决定）。"
                  ).pack(anchor=tk.W)

    def _create_param_page(self, notebook: ttk.Notebook):
        """参数页：传统链滑动条（神经链下禁用）。"""
        page = ttk.Frame(notebook, padding=8)
        notebook.add(page, text="参数")

        self.detail_var = tk.DoubleVar(value=self.params.detail)
        self.line_sensitivity_var = tk.DoubleVar(
            value=self.params.line_sensitivity)
        self.coherence_var = tk.DoubleVar(value=self.params.coherence)
        self.thickness_var = tk.DoubleVar(value=self.params.thickness)
        self.denoise_var = tk.DoubleVar(value=self.params.denoise)
        self.illumination_var = tk.DoubleVar(value=self.params.illumination)
        self.smooth_radius_var = tk.DoubleVar(
            value=float(self.params.smooth_radius))
        self.min_area_var = tk.DoubleVar(value=float(self.params.min_area))
        self.max_side_var = tk.DoubleVar(value=float(self.params.max_side))

        for label, var, frm, to in (
                ("细节强度 detail", self.detail_var, 0.0, 1.0),
                ("灵敏度 sensitivity", self.line_sensitivity_var, 0.0, 1.0),
                ("连贯性 coherence", self.coherence_var, 0.0, 1.0),
                ("线条粗细 thickness", self.thickness_var, 0.5, 3.0),
                ("去噪 denoise", self.denoise_var, 0.0, 1.0),
                ("光照归一 illumination", self.illumination_var, 0.0, 1.0),
                ("保边半径 smooth", self.smooth_radius_var, 1, 15),
                ("最小连通域 min_area", self.min_area_var, 0, 100),
                ("最长边上限 max_side", self.max_side_var, 512, 8192),
        ):
            self._add_slider(page, label, var, frm, to)

        self.traditional_hint = ttk.Label(
            page, foreground="#888", wraplength=240, justify=tk.LEFT,
            text="以上参数仅传统语义（fdog/xdog/adaptive）生效；"
                 "神经语义下自动禁用。改参后点「▶ 转换」生效。")
        self.traditional_hint.pack(anchor=tk.W, pady=(6, 0))
        self._sync_traditional_state()

    def _add_slider(self, parent, label: str, var, frm: float, to: float):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        scale = ttk.Scale(row, from_=frm, to=to, variable=var,
                          command=lambda _v: self.on_param_change())
        scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._sliders.append(scale)

    def _sync_traditional_state(self):
        """神经链下禁用传统参数滑动条（对输出零影响，防误导）。"""
        neural = self.semantic_var.get() in NEURAL_SEMANTICS
        for w in self._sliders:
            try:
                w.state(["disabled"] if neural else ["normal"])
            except tk.TclError:
                pass

    # ------------------------------------------------------------ 事件绑定

    def _bind_events(self):
        self.canvas.bind("<ButtonPress-1>", self._pan_start)
        self.canvas.bind("<B1-Motion>", self._pan_move)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.root.bind("<Configure>", self.on_resize)
        # 快捷键：Ctrl+O 打开 / Ctrl+Enter 转换 / Ctrl+S 导出
        self.root.bind("<Control-o>", lambda _e: self.open_image())
        self.root.bind("<Control-Return>", lambda _e: self.convert_now())
        self.root.bind("<Control-s>", lambda _e: self.export_image())

    def _pan_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _pan_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mousewheel(self, event):
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self.zoom_scale *= factor
        self._apply_zoom()

    def on_resize(self, event):
        if event.widget is self.root and self.original_img is not None:
            self._apply_zoom()

    # ------------------------------------------------------------ 打开图像

    def open_image(self):
        filetypes = [
            ("图片/视频", "*.png *.jpg *.jpeg *.bmp *.webp *.gif "
                          "*.mp4 *.avi *.mov *.mkv"),
            ("所有文件", "*.*"),
        ]
        path = filedialog.askopenfilename(title="打开图片", filetypes=filetypes)
        if not path:
            return
        if is_supported_video(path):
            self._load_video_first_frame(path)
        else:
            img = load_image(path)
            if img is None:
                messagebox.showerror("错误", f"无法读取: {path}")
                return
            self.original_img = img
        self.input_path = path
        self.processed_img = None
        self.neural_map = None
        self.dirty = True
        self.zoom_fit()
        self.update_preview()
        h, w = self.original_img.shape[:2]
        self.status_var.set(
            f"{os.path.basename(path)} 已打开（{w}×{h}）— 点「▶ 转换」生成线稿")

    def _load_video_first_frame(self, path: str):
        cap = cv2.VideoCapture(path)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            messagebox.showerror("错误", f"无法读取视频首帧: {path}")
            return
        self.original_img = frame
        self.status_var.set(f"视频首帧已加载（{path}）")

    # ------------------------------------------------------------ 转换（后台线程）

    def convert_now(self):
        """显式转换按钮：后台线程跑管线，UI 不冻结（v1.27.0 重构核心）。

        大图神经 ensemble 可达数十秒——旧版滑动条自动重跑会冻结
        整个窗口，用户以为崩溃。"""
        if self.original_img is None:
            messagebox.showwarning("提示", "请先打开图片")
            return
        if self.converting:
            return
        self._update_params_from_ui()
        self.converting = True
        self._enable_buttons(False)
        semantic = self.params.semantic
        self.status_var.set(f"转换中（{semantic}）… "
                            "神经语义大图可能需要数十秒")
        self.btn_convert.configure(text="转换中…")

        img = self.original_img.copy()
        params = LineArtParams(**vars(self.params))
        neural_map = self.neural_map

        def work():
            t0 = time.perf_counter()
            try:
                result = to_lineart(img, params, semantic_map=neural_map)
                self._result_q.put(
                    ("ok", (result, time.perf_counter() - t0)))
            except Exception as e:
                self._result_q.put(("err", repr(e)))

        threading.Thread(target=work, daemon=True,
                         name="lineart-convert").start()

    def _poll_results(self):
        """主线程轮询后台结果队列（100ms 周期）。"""
        try:
            while True:
                kind, payload = self._result_q.get_nowait()
                if kind == "ok":
                    result, dt = payload
                    self.processed_img = result
                    self.dirty = False
                    self._enable_buttons(True)
                    self.btn_convert.configure(text="▶ 转换")
                    self.converting = False
                    mode = ("ensemble 三语义融合" if
                            self.params.semantic == "ensemble"
                            else self.params.semantic)
                    h, w = self.original_img.shape[:2]
                    self.status_var.set(
                        f"转换完成（{mode}，{dt:.1f}s，{w}×{h}）")
                    self.update_preview()
                elif kind == "err":
                    self._enable_buttons(True)
                    self.btn_convert.configure(text="▶ 转换")
                    self.converting = False
                    self.status_var.set("转换失败")
                    messagebox.showerror("转换错误", str(payload))
                elif kind == "selfcheck":
                    ok, report = payload
                    self._enable_buttons(True)
                    self.btn_selfcheck.configure(text="运行自检")
                    self.status_var.set("自检通过 ✓" if ok else "自检失败 ✗")
                    messagebox.showinfo(
                        "自检通过" if ok else "自检失败", report)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_results)

    def _enable_buttons(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        for btn in (self.btn_open, self.btn_convert, self.btn_export,
                    self.btn_selfcheck):
            btn.configure(state=state)

    # ------------------------------------------------------------ 自检（后台线程）

    def run_selfcheck(self):
        """GUI 内运行诊断自检（复用 CLI --selfcheck 同一入口）。"""
        if self.converting:
            return
        self._enable_buttons(False)
        self.btn_selfcheck.configure(text="自检中…")
        self.status_var.set("自检中：端到端真实转换检测…")

        def work():
            try:
                from ..selfcheck import run_selfcheck
                ok, results = run_selfcheck()
                lines = ["lineart-precise 功能自检", "=" * 56]
                for r in results:
                    lines.append(f"  [{r.status:<4}] {r.name:<22} "
                                 f"{r.detail}  ({r.ms}ms)")
                core = [r for r in results if r.category == "core"]
                lines.append("-" * 56)
                lines.append(f"核心 {len(core)} 项: "
                             f"PASS {sum(r.status == 'PASS' for r in core)}"
                             f" / FAIL {sum(r.status == 'FAIL' for r in core)}")
                lines.append("结论: " + ("全部核心功能可用" if ok
                                         else "存在失败项，详见上方"))
                self._result_q.put(("selfcheck", (ok, "\n".join(lines))))
            except Exception as e:
                self._result_q.put(("selfcheck", (False, f"自检异常: {e!r}")))

        threading.Thread(target=work, daemon=True,
                         name="lineart-selfcheck").start()

    # ------------------------------------------------------------ 参数

    def on_param_change(self):
        """参数变动：不自动重转换（大图/神经太慢），只标记待转换。"""
        self.dirty = True
        self._sync_traditional_state()
        if self.original_img is not None and not self.converting:
            self.status_var.set("参数已修改 — 点「▶ 转换」生效")

    def _update_params_from_ui(self):
        semantic = self.semantic_var.get()
        self.params.semantic = semantic
        self.params.scene = next(
            v for v, t in SCENE_CHOICES if t == self.scene_var.get())
        self.params.output = next(
            v for v, t in OUTPUT_CHOICES if t == self.output_var.get())
        self.params.dexined_scale = next(
            v for v, t in DEX_SCALE_CHOICES if t == self.dex_scale_var.get())
        if semantic not in NEURAL_SEMANTICS:
            self.params.detail = self.detail_var.get()
            self.params.line_sensitivity = self.line_sensitivity_var.get()
            self.params.coherence = self.coherence_var.get()
            self.params.thickness = self.thickness_var.get()
            self.params.denoise = self.denoise_var.get()
            self.params.illumination = self.illumination_var.get()
            self.params.smooth_radius = int(self.smooth_radius_var.get())
            self.params.min_area = int(self.min_area_var.get())
            self.params.max_side = int(self.max_side_var.get())

    # ------------------------------------------------------------ 预览显示

    def update_preview(self):
        if self.original_img is None:
            self.canvas.delete("all")
            self.canvas.create_text(
                max(self.canvas.winfo_width() // 2, 400),
                max(self.canvas.winfo_height() // 2, 300),
                text="打开图片开始（Ctrl+O）", fill="gray", font=("", 16))
            return
        display = self._get_display_image()
        if display is not None:
            self._display_on_canvas(display)

    def _get_display_image(self) -> Optional[np.ndarray]:
        if self.preview_mode.get() == "original":
            return self.original_img.copy()
        if self.processed_img is None:
            return None
        return self.processed_img.copy()

    def _display_on_canvas(self, img: np.ndarray):
        h, w = img.shape[:2]
        if self.zoom_scale <= 0:
            self.zoom_scale = 1.0
        dw = max(1, int(w * self.zoom_scale))
        dh = max(1, int(h * self.zoom_scale))
        interp = cv2.INTER_AREA if dw < w else cv2.INTER_LINEAR
        disp = cv2.resize(img, (dw, dh), interpolation=interp) \
            if (dw, dh) != (w, h) else img
        if disp.ndim == 2:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(rgb)
        self._tk_photo = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_photo)

    def zoom_fit(self):
        if self.original_img is None:
            return
        h, w = self.original_img.shape[:2]
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        self._fit_scale = min(cw / w, ch / h)
        self.zoom_scale = self._fit_scale
        self._apply_zoom()

    def zoom_actual(self):
        self.zoom_scale = 1.0
        self._apply_zoom()

    def _apply_zoom(self):
        if self.original_img is not None:
            self.update_preview()

    # ------------------------------------------------------------ 导出

    def export_image(self):
        if self.processed_img is None:
            messagebox.showwarning("提示", "请先点「▶ 转换」生成线稿")
            return
        if self.input_path is None:
            return
        stem = os.path.splitext(os.path.basename(self.input_path))[0]
        filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"),
                     ("BMP", "*.bmp"), ("所有文件", "*.*")]
        path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=filetypes,
            initialfile=f"{stem}_lineart.png")
        if not path:
            return
        try:
            save_image(path, self.processed_img)
            self.status_var.set(f"已导出: {os.path.basename(path)}")
            messagebox.showinfo("成功", f"已保存到: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")


def main():
    """启动 GUI"""
    root = tk.Tk()
    LineArtPreviewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
