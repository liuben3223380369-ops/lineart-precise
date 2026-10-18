# -*- coding: utf-8 -*-
"""文件名: preview.py
描述: 实时预览 GUI（tkinter 实现），支持参数调整、对比预览和导出
"""
import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Tuple

import cv2
import numpy as np

from ..params import LineArtParams
from ..pipeline import to_lineart
from ..io import load_image, save_image, is_supported_video
from ..io.video import process_video, process_gif


class LineArtPreviewGUI:
    """线稿实时预览 GUI"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LineArt Precise - 实时预览")
        self.root.geometry("1200x800")
        
        # 当前图像
        self.input_path: Optional[str] = None
        self.original_img: Optional[np.ndarray] = None
        self.processed_img: Optional[np.ndarray] = None
        
        # 参数
        self.params = LineArtParams()
        
        # 预览模式
        self.preview_mode = tk.StringVar(value="sidebyside")  # sidebyside, original, processed
        
        # 缩放比例
        self.zoom_scale = 1.0
        
        # 创建 UI
        self._create_widgets()
        
        # 绑定事件
        self._bind_events()
    
    def _create_widgets(self):
        """创建所有 UI 组件"""
        # 主框架
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        control_frame = ttk.LabelFrame(main_frame, text="参数控制", padding=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # 文件操作按钮
        ttk.Button(control_frame, text="打开图片", command=self.open_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="导出结果", command=self.export_image).pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="保存参数", command=self.save_params).pack(fill=tk.X, pady=2)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 预览模式选择
        mode_frame = ttk.LabelFrame(control_frame, text="预览模式", padding=5)
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(mode_frame, text="对比预览", variable=self.preview_mode, 
                       value="sidebyside", command=self.update_preview).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="原图", variable=self.preview_mode, 
                       value="original", command=self.update_preview).pack(anchor=tk.W)
        ttk.Radiobutton(mode_frame, text="线稿", variable=self.preview_mode, 
                       value="processed", command=self.update_preview).pack(anchor=tk.W)
        
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 参数滑动条
        self._create_param_sliders(control_frame)
        
        # 右侧预览区域
        preview_frame = ttk.LabelFrame(main_frame, text="预览", padding=5)
        preview_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 画布
        canvas_frame = ttk.Frame(preview_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # 使用 Canvas 显示图像
        self.canvas = tk.Canvas(canvas_frame, bg="white", highlightthickness=1, 
                                highlightbackground="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _create_param_sliders(self, parent):
        """创建参数滑动条"""
        # 使用 Notebook 分页
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 基础参数页
        basic_frame = ttk.Frame(notebook)
        notebook.add(basic_frame, text="基础")
        
        # 细节参数
        self._add_slider(basic_frame, "细节强度", "detail", 0.0, 1.0, 0.6, 0.05)
        # 灵敏度
        self._add_slider(basic_frame, "线条灵敏度", "line_sensitivity", 0.0, 1.0, 0.5, 0.05)
        # 连贯性
        self._add_slider(basic_frame, "线条连贯", "coherence", 0.0, 1.0, 0.6, 0.05)
        # 粗细
        self._add_slider(basic_frame, "线条粗细", "thickness", 0.5, 5.0, 1.0, 0.1)
        
        # 预处理页
        preprocess_frame = ttk.Frame(notebook)
        notebook.add(preprocess_frame, text="预处理")
        
        # 去噪
        self._add_slider(preprocess_frame, "去噪强度", "denoise", 0.0, 1.0, 0.4, 0.05)
        # 光照
        self._add_slider(preprocess_frame, "光照归一化", "illumination", 0.0, 1.0, 0.8, 0.05)
        # 锐化
        self._add_slider(preprocess_frame, "冲击锐化", "sharpen", 0.0, 1.0, 0.0, 0.05)
        
        # 高级页
        advanced_frame = ttk.Frame(notebook)
        notebook.add(advanced_frame, text="高级")
        
        # 平滑半径
        self._add_slider(advanced_frame, "平滑半径", "smooth_radius", 1, 15, 3, 1)
        # 最小面积
        self._add_slider(advanced_frame, "最小连通域", "min_area", 0, 100, 10, 5)
        # 最大边长
        self._add_slider(advanced_frame, "最大边长", "max_side", 512, 8192, 4096, 128)
        
        # 语义选择
        semantic_frame = ttk.LabelFrame(advanced_frame, text="语义选择")
        semantic_frame.pack(fill=tk.X, pady=5)
        self.semantic_var = tk.StringVar(value="fdog")
        ttk.Radiobutton(semantic_frame, text="FDoG 流线", variable=self.semantic_var, 
                       value="fdog", command=self.on_param_change).pack(anchor=tk.W)
        ttk.Radiobutton(semantic_frame, text="XDoG 多尺度", variable=self.semantic_var, 
                       value="xdog", command=self.on_param_change).pack(anchor=tk.W)
        ttk.Radiobutton(semantic_frame, text="自适应阈值", variable=self.semantic_var, 
                       value="adaptive", command=self.on_param_change).pack(anchor=tk.W)
        
        # 输出模式
        output_frame = ttk.LabelFrame(advanced_frame, text="输出模式")
        output_frame.pack(fill=tk.X, pady=5)
        self.output_var = tk.StringVar(value="grayscale")
        ttk.Radiobutton(output_frame, text="白底黑线", variable=self.output_var, 
                       value="grayscale", command=self.on_param_change).pack(anchor=tk.W)
        ttk.Radiobutton(output_frame, text="纯黑白", variable=self.output_var, 
                       value="binary", command=self.on_param_change).pack(anchor=tk.W)
        ttk.Radiobutton(output_frame, text="彩色线", variable=self.output_var, 
                       value="color", command=self.on_param_change).pack(anchor=tk.W)
    
    def _add_slider(self, parent, label: str, param_name: str, 
                    from_: float, to: float, initial: float, resolution: float):
        """添加一个滑动条"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=12).pack(side=tk.LEFT)
        
        var = tk.DoubleVar(value=initial)
        slider = ttk.Scale(frame, from_=from_, to=to, variable=var, 
                          orient=tk.HORIZONTAL, length=150)
        slider.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        
        value_label = ttk.Label(frame, text=f"{initial:.2f}", width=6)
        value_label.pack(side=tk.RIGHT)
        
        # 存储引用
        self.__dict__[f"{param_name}_var"] = var
        self.__dict__[f"{param_name}_slider"] = slider
        self.__dict__[f"{param_name}_label"] = value_label
        
        # 绑定事件
        slider.bind("<Motion>", lambda e, p=param_name, l=value_label: self._update_slider_label(p, l))
        slider.bind("<ButtonRelease-1>", lambda e: self.on_param_change())
    
    def _update_slider_label(self, param_name: str, label: ttk.Label):
        """更新滑动条标签"""
        var = self.__dict__[f"{param_name}_var"]
        label.config(text=f"{var.get():.2f}")
    
    def _bind_events(self):
        """绑定事件"""
        # 鼠标滚轮缩放
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        # 窗口大小改变
        self.root.bind("<Configure>", self.on_resize)
    
    def open_image(self):
        """打开图片文件"""
        filetypes = [
            ("所有支持的格式", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff *.mp4 *.avi *.mov *.gif"),
            ("图片", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
            ("视频", "*.mp4 *.avi *.mov"),
            ("动图", "*.gif"),
            ("所有文件", "*.*")
        ]
        path = filedialog.askopenfilename(filetypes=filetypes)
        if not path:
            return
        
        self.input_path = path
        self.status_var.set(f"加载中: {os.path.basename(path)}")
        self.root.update()
        
        try:
            if is_supported_video(path):
                # 视频/动图：提取第一帧
                self._load_video_first_frame(path)
            else:
                # 图片
                self.original_img = load_image(path)
            
            self.update_preview()
            self.status_var.set(f"已加载: {os.path.basename(path)} ({self.original_img.shape[1]}x{self.original_img.shape[0]})")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")
            self.status_var.set("加载失败")
    
    def _load_video_first_frame(self, path: str):
        """加载视频的第一帧"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise IOError(f"无法打开视频: {path}")
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise IOError(f"无法读取视频帧: {path}")
        
        self.original_img = frame
    
    def on_param_change(self):
        """参数改变时重新处理"""
        if self.original_img is None:
            return
        
        # 更新参数
        self._update_params_from_ui()
        
        # 处理
        self.process_and_preview()
    
    def _update_params_from_ui(self):
        """从 UI 更新参数"""
        self.params.detail = self.detail_var.get()
        self.params.line_sensitivity = self.line_sensitivity_var.get()
        self.params.coherence = self.coherence_var.get()
        self.params.thickness = self.thickness_var.get()
        self.params.denoise = self.denoise_var.get()
        self.params.illumination = self.illumination_var.get()
        self.params.sharpen = self.sharpen_var.get()
        self.params.smooth_radius = int(self.smooth_radius_var.get())
        self.params.min_area = int(self.min_area_var.get())
        self.params.max_side = int(self.max_side_var.get())
        self.params.semantic = self.semantic_var.get()
        self.params.output = self.output_var.get()
    
    def process_and_preview(self):
        """处理图像并更新预览"""
        if self.original_img is None:
            return
        
        try:
            self.processed_img = to_lineart(self.original_img, self.params)
            self.update_preview()
        except Exception as e:
            messagebox.showerror("处理错误", f"处理失败: {str(e)}")
    
    def update_preview(self):
        """更新预览画布"""
        if self.original_img is None:
            self.canvas.delete("all")
            self.canvas.create_text(400, 400, text="打开图片开始预览", fill="gray", font=("", 16))
            return
        
        # 获取显示图像
        display_img = self._get_display_image()
        if display_img is None:
            return
        
        # 转换为 PIL 格式用于显示
        self._display_on_canvas(display_img)
    
    def _get_display_image(self) -> Optional[np.ndarray]:
        """获取根据预览模式显示的图像"""
        mode = self.preview_mode.get()
        
        if mode == "original":
            # 原图
            return self.original_img.copy()
        elif mode == "processed":
            # 线稿
            if self.processed_img is None:
                return None
            return self.processed_img.copy()
        else:
            # 对比预览
            if self.processed_img is None:
                return self.original_img.copy()
            
            # 并排显示
            h1, w1 = self.original_img.shape[:2]
            h2, w2 = self.processed_img.shape[:2]
            
            # 统一高度
            h = max(h1, h2)
            
            # 调整原图高度
            orig_resized = cv2.resize(self.original_img, (int(w1 * h / h1), h))
            
            # 调整线稿高度
            if self.processed_img.ndim == 2:
                proc_bgr = cv2.cvtColor(self.processed_img, cv2.COLOR_GRAY2BGR)
            else:
                proc_bgr = self.processed_img
            proc_resized = cv2.resize(proc_bgr, (int(w2 * h / h2), h))
            
            # 合并
            combined = np.hstack([orig_resized, proc_resized])
            return combined
    
    def _display_on_canvas(self, img: np.ndarray):
        """在 Canvas 上显示图像"""
        # 获取画布尺寸
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        
        if canvas_w <= 1 or canvas_h <= 1:
            return
        
        # 计算缩放
        img_h, img_w = img.shape[:2]
        scale = min(canvas_w / img_w, canvas_h / img_h) * self.zoom_scale
        scale = min(scale, 10.0)  # 限制最大缩放
        
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        
        # 调整尺寸
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 RGB
        if resized.ndim == 2:
            rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 创建 PhotoImage
        from PIL import Image, ImageTk
        pil_img = Image.fromarray(rgb)
        self.photo = ImageTk.PhotoImage(pil_img)
        
        # 显示
        self.canvas.delete("all")
        x = (canvas_w - new_w) // 2
        y = (canvas_h - new_h) // 2
        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        
        # 显示尺寸信息
        self.canvas.create_text(10, 10, text=f"{img_w}x{img_h} | 缩放: {scale*100:.0f}%", 
                                anchor=tk.NW, fill="white", font=("", 10))
    
    def on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if event.delta > 0:
            self.zoom_scale = min(self.zoom_scale * 1.2, 5.0)
        else:
            self.zoom_scale = max(self.zoom_scale / 1.2, 0.1)
        self.update_preview()
    
    def on_resize(self, event):
        """窗口大小改变"""
        if self.original_img is not None:
            self.update_preview()
    
    def export_image(self):
        """导出结果"""
        if self.processed_img is None:
            messagebox.showwarning("提示", "请先打开图片并处理")
            return
        
        if self.input_path is None:
            return
        
        # 生成默认输出路径
        stem = os.path.splitext(os.path.basename(self.input_path))[0]
        default_name = f"{stem}_lineart.png"
        
        filetypes = [
            ("PNG", "*.png"),
            ("JPEG", "*.jpg *.jpeg"),
            ("BMP", "*.bmp"),
            ("所有文件", "*.*")
        ]
        
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                            filetypes=filetypes,
                                            initialfile=default_name)
        if not path:
            return
        
        try:
            save_image(path, self.processed_img)
            self.status_var.set(f"已导出: {os.path.basename(path)}")
            messagebox.showinfo("成功", f"已保存到: {path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def save_params(self):
        """保存参数"""
        filetypes = [("JSON", "*.json"), ("所有文件", "*.*")]
        path = filedialog.asksaveasfilename(defaultextension=".json",
                                            filetypes=filetypes)
        if not path:
            return
        
        try:
            import json
            params_dict = {
                "detail": self.params.detail,
                "line_sensitivity": self.params.line_sensitivity,
                "coherence": self.params.coherence,
                "thickness": self.params.thickness,
                "denoise": self.params.denoise,
                "illumination": self.params.illumination,
                "sharpen": self.params.sharpen,
                "smooth_radius": self.params.smooth_radius,
                "min_area": self.params.min_area,
                "max_side": self.params.max_side,
                "semantic": self.params.semantic,
                "output": self.params.output,
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(params_dict, f, indent=2, ensure_ascii=False)
            self.status_var.set(f"参数已保存: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")


def main():
    """启动 GUI"""
    root = tk.Tk()
    app = LineArtPreviewGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
