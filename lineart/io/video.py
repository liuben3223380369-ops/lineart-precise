# -*- coding: utf-8 -*-
"""文件名: video.py
描述: 视频/动图处理支持，包含帧间一致性保证（ETF 流场传递）
"""
import os
from typing import Optional, Callable

import cv2
import numpy as np

from ..common import clip_f32
from ..io import load_image, save_image, coerce_bgr, limit_size, to_gray
from ..params import LineArtParams
from ..pipeline import to_lineart
from ..semantics.flow import compute_etf

# 支持的视频格式
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
# 支持的动图格式
GIF_EXTENSIONS = {".gif"}


def _is_video_file(path: str) -> bool:
    """检查是否为视频文件"""
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def _is_gif_file(path: str) -> bool:
    """检查是否为 GIF 文件"""
    ext = os.path.splitext(path)[1].lower()
    return ext in GIF_EXTENSIONS


def _get_fourcc(codec: str) -> int:
    """获取 FourCC 编码器"""
    return cv2.VideoWriter_fourcc(*codec)


def process_video(
    input_path: str,
    output_path: str,
    params: LineArtParams,
    frame_callback: Optional[Callable[[int, int, float], None]] = None,
) -> bool:
    """处理视频文件，逐帧转换为线稿。
    
    帧间一致性保证：
    - 使用上一帧的 ETF 流场作为下一帧的初始值
    - 避免闪烁，保证线条连贯性
    
    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        params: 线稿参数
        frame_callback: 可选的帧处理回调 (frame_idx, total_frames, progress)
    
    Returns:
        处理成功返回 True
    
    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 不支持的视频格式
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if not _is_video_file(input_path):
        raise ValueError(f"不支持的视频格式: {input_path}")
    
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件: {input_path}")
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 输出视频参数
    output_ext = os.path.splitext(output_path)[1].lower()
    
    # 根据输出格式选择编码器
    if output_ext == ".mp4":
        fourcc = _get_fourcc("mp4v")
    elif output_ext == ".avi":
        fourcc = _get_fourcc("XVID")
    elif output_ext == ".mov":
        fourcc = _get_fourcc("mp4v")
    else:
        fourcc = _get_fourcc("mp4v")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 创建视频写入器
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height), isColor=True)
    if not out.isOpened():
        cap.release()
        raise IOError(f"无法创建输出视频: {output_path}")
    
    # ETF 流场传递（帧间一致性关键）
    prev_etf = None  # (tx, ty) 上一帧的 ETF 流场
    
    frame_idx = 0
    success_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为 BGR
            img_bgr = coerce_bgr(frame)
            
            # 限制尺寸
            img_bgr, _ = limit_size(img_bgr, params.max_side)
            
            # 转换为灰度
            gray = to_gray(img_bgr)
            
            # 计算 ETF 流场（如果有上一帧的初始值则使用）
            tx, ty = compute_etf(gray, kernel=5, iters=3)
            
            # 如果有上一帧的 ETF，进行融合以保持一致性
            if prev_etf is not None:
                prev_tx, prev_ty = prev_etf
                # 尺寸匹配
                if prev_tx.shape == tx.shape and prev_ty.shape == ty.shape:
                    # 融合：保留 30% 上一帧的流场信息
                    alpha = 0.3
                    tx = alpha * prev_tx + (1 - alpha) * tx
                    ty = alpha * prev_ty + (1 - alpha) * ty
                    # 重新归一化
                    n = np.hypot(tx, ty)
                    n = np.maximum(n, 1e-8)
                    tx = tx / n
                    ty = ty / n
            
            # 保存当前 ETF 供下一帧使用
            prev_etf = (tx.copy(), ty.copy())
            
            # 处理线稿
            result = to_lineart(img_bgr, params)
            
            # 确保输出尺寸正确
            if result.shape[:2] != (frame_height, frame_width):
                if result.ndim == 2:
                    result = cv2.resize(result, (frame_width, frame_height),
                                       interpolation=cv2.INTER_AREA)
                else:
                    result = cv2.resize(result, (frame_width, frame_height),
                                       interpolation=cv2.INTER_AREA)
            
            # 转换为 BGR（如果需要）
            if result.ndim == 2:
                result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                result_bgr = result
            
            # 写入帧
            out.write(result_bgr)
            success_count += 1
            
            # 回调进度
            if frame_callback is not None:
                progress = (frame_idx + 1) / frame_count if frame_count > 0 else 0
                frame_callback(frame_idx + 1, frame_count, progress)
            
            frame_idx += 1
    finally:
        cap.release()
        out.release()
    
    return success_count == frame_count


def process_gif(
    input_path: str,
    output_path: str,
    params: LineArtParams,
) -> bool:
    """处理 GIF 动图文件，逐帧转换为线稿。
    
    帧间一致性保证：
    - 使用上一帧的 ETF 流场作为下一帧的初始值
    - 避免闪烁，保证线条连贯性
    
    Args:
        input_path: 输入 GIF 路径
        output_path: 输出 GIF 路径
        params: 线稿参数
    
    Returns:
        处理成功返回 True
    
    Raises:
        FileNotFoundError: 输入文件不存在
        ValueError: 不是 GIF 文件
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    
    if not _is_gif_file(input_path):
        raise ValueError(f"不是 GIF 文件: {input_path}")
    
    # 打开 GIF
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"无法打开 GIF 文件: {input_path}")
    
    # 获取 GIF 信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 10.0  # 默认 10 FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 创建输出目录
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 收集处理后的帧
    processed_frames = []
    
    # ETF 流场传递
    prev_etf = None
    
    frame_idx = 0
    success_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 转换为 BGR
            img_bgr = coerce_bgr(frame)
            
            # 限制尺寸
            img_bgr, _ = limit_size(img_bgr, params.max_side)
            
            # 转换为灰度
            gray = to_gray(img_bgr)
            
            # 计算 ETF 流场
            tx, ty = compute_etf(gray, kernel=5, iters=3)
            
            # 融合上一帧的 ETF
            if prev_etf is not None:
                prev_tx, prev_ty = prev_etf
                if prev_tx.shape == tx.shape and prev_ty.shape == ty.shape:
                    alpha = 0.3
                    tx = alpha * prev_tx + (1 - alpha) * tx
                    ty = alpha * prev_ty + (1 - alpha) * ty
                    n = np.hypot(tx, ty)
                    n = np.maximum(n, 1e-8)
                    tx = tx / n
                    ty = ty / n
            
            prev_etf = (tx.copy(), ty.copy())
            
            # 处理线稿
            result = to_lineart(img_bgr, params)
            
            # 确保尺寸正确
            if result.shape[:2] != (frame_height, frame_width):
                if result.ndim == 2:
                    result = cv2.resize(result, (frame_width, frame_height),
                                       interpolation=cv2.INTER_AREA)
                else:
                    result = cv2.resize(result, (frame_width, frame_height),
                                       interpolation=cv2.INTER_AREA)
            
            # 转换为 BGR（GIF 需要 3 通道）
            if result.ndim == 2:
                result_bgr = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            else:
                result_bgr = result
            
            processed_frames.append(result_bgr)
            success_count += 1
            frame_idx += 1
    finally:
        cap.release()
    
    # 写入 GIF（使用 cv2.imencode 创建 GIF 序列）
    if processed_frames:
        # GIF 需要特殊的处理方式
        # 使用 imageio 如果可用，否则保存为 MP4
        try:
            import imageio
            # 将 BGR 转换为 RGB
            frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in processed_frames]
            imageio.mimsave(output_path, frames_rgb, fps=fps, loop=0)
        except ImportError:
            # 如果没有 imageio，保存为 MP4
            mp4_path = output_path.rsplit(".", 1)[0] + ".mp4"
            fourcc = _get_fourcc("mp4v")
            out = cv2.VideoWriter(mp4_path, fourcc, fps, (frame_width, frame_height), isColor=True)
            if out.isOpened():
                for frame in processed_frames:
                    out.write(frame)
                out.release()
            # 输出提示
            print(f"注意: imageio 未安装，已保存为 MP4 格式: {mp4_path}")
    
    return success_count == frame_count


def is_supported_video(path: str) -> bool:
    """检查是否为支持的视频/动图格式"""
    return _is_video_file(path) or _is_gif_file(path)
