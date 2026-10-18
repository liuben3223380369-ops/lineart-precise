"""图像读写：支持中文路径与 Alpha 通道合成到白底。"""
import cv2
import numpy as np

MAX_SIDE = 4096


def coerce_bgr(img):
    """标准化输入为 uint8 BGR 3 通道（宽容各种 dtype/通道/灰度输入）。

    处理规则：
    - None / 维度非 2D-3D / 空尺寸 → 明确 ValueError
    - float64/float32 → 按值域归一化到 uint8（0~1 → *255，否则 clip）
    - 其他 dtype → astype(uint8)
    - 灰度(2D 或 单通道3D) → GRAY2BGR
    - RGBA(4 通道) → alpha 合成白底
    - BGR(3 通道) → 原样
    """
    if img is None:
        raise ValueError("输入图像为 None")
    arr = np.asarray(img)
    if arr.ndim not in (2, 3):
        raise ValueError(f"输入图像维度必须是 2 或 3，得到 {arr.ndim}")
    h, w = arr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("输入图像为空（0 尺寸）")

    if arr.dtype == np.float64:
        arr = arr.astype(np.float32)
    if arr.dtype == np.float32:
        vmax, vmin = float(arr.max()), float(arr.min())
        if 0.0 <= vmin and vmax <= 1.0:
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    c = arr.shape[2]
    if c == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    if c == 3:
        return arr
    if c == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32)
        return (rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    raise ValueError(f"输入图像通道数必须是 1/3/4，得到 {c}")


def load_image(path):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {path}")
    return coerce_bgr(img)


def limit_size(img, max_side=MAX_SIDE):
    h, w = img.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return img, 1.0
    scale = max_side / float(side)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                     interpolation=cv2.INTER_AREA)
    return img, scale


def save_image(path, img):
    ext = str(path).rsplit(".", 1)[-1].lower()
    if ext not in ("png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"):
        ext = "png"  # 无扩展名/未知扩展时兜底 png
    ok, buf = cv2.imencode(f".{ext}", img)
    if not ok:
        raise IOError(f"无法编码图像: {path}")
    buf.tofile(str(path))


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
