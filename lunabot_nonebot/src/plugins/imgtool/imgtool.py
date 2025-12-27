import numpy as np
from PIL import Image
from typing import List, Union
from dataclasses import dataclass
# --- 底层 NumPy 算法实现 ---

def _color_diff_sq(a, b):
    return np.sum((a[..., :3].astype(np.int32) - b[..., :3].astype(np.int32))**2, axis=-1)

def _floodfill_numpy(frame, sy, sx, max_color, tolerance_sq):
    h, w = frame.shape[:2]
    dst_color = np.array([0, 0, 0, 0], dtype=np.uint8)
    stack = [(sy, sx)]
    
    # 起点检查
    if frame[sy, sx, 3] == 0 or _color_diff_sq(frame[sy, sx], max_color) > tolerance_sq:
        return

    frame[sy, sx] = dst_color
    while stack:
        y, x = stack.pop()
        for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                pixel = frame[ny, nx]
                if pixel[3] != 0 and _color_diff_sq(pixel, max_color) <= tolerance_sq:
                    frame[ny, nx] = dst_color
                    stack.append((ny, nx))

def _cutout_logic(img_array: np.ndarray, tolerance: int) -> tuple[np.ndarray, dict]:
    n, h, w, _ = img_array.shape
    tolerance_sq = (tolerance * tolerance) * 3
    
    # 获取第一帧边缘最频繁的颜色
    frame_0 = img_array[0]
    edges = np.concatenate([
        frame_0[0, :], frame_0[-1, :], frame_0[:, 0], frame_0[:, -1]
    ])
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    max_color = unique[np.argmax(counts)]

    for t in range(n):
        # 仅从边缘触发泛洪
        for y in [0, h-1]:
            for x in range(w):
                _floodfill_numpy(img_array[t], y, x, max_color, tolerance_sq)
        for x in [0, w-1]:
            for y in range(h):
                _floodfill_numpy(img_array[t], y, x, max_color, tolerance_sq)
    return img_array, {}

def _shrink_logic(img_array: np.ndarray, alpha_threshold: int, edge: int) -> tuple[np.ndarray, dict]:
    coords = np.where(img_array[..., 3] > alpha_threshold)
    if coords[0].size == 0: return np.zeros((1, 1, 1, 4), dtype=np.uint8)
    
    t_min, t_max = coords[0].min(), coords[0].max() + 1
    y_min, y_max = coords[1].min(), coords[1].max() + 1
    x_min, x_max = coords[2].min(), coords[2].max() + 1
    
    # bx, by, bw, bh
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

    # 计算新尺寸并执行 Padding
    content = img_array[t_min:t_max, y_min:y_max, x_min:x_max]
    return np.pad(content, ((0,0), (edge,edge), (edge,edge), (0,0)), mode='constant'), {'bbox': bbox}

# --- 业务调用接口 ---
@dataclass
class ImageToolResult:
    image: Image.Image | List[Image.Image]
    extra_info: dict

def execute_imgtool_py(
    image: Image.Image | List[Image.Image], 
    command: str, 
    *args
) -> ImageToolResult:
    """
    直接输入图像并处理
    """
    is_single_frame = isinstance(image, Image.Image)
    if is_single_frame:
        image = [image]
    # 转换为 NumPy 数组 (n, h, w, 4)
    img_array = np.array([np.array(img.convert('RGBA')) for img in image])
    
    # 3. 根据命令处理
    if command == "cutout":
        tolerance = int(args[0])
        processed, extra_info = _cutout_logic(img_array, tolerance)
    elif command == "shrink":
        threshold, edge_size = int(args[0]), int(args[1])
        processed, extra_info = _shrink_logic(img_array, threshold, edge_size)
    else:
        raise ValueError(f"未知命令: {command}")
    
    # 4. 转回 PIL 对象
    result_images = [Image.fromarray(f) for f in processed]
    output = ImageToolResult(
        image=result_images[0] if is_single_frame else result_images,
        extra_info=extra_info
    )
    return output