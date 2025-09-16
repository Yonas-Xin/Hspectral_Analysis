import pywt
import numpy as np
import cv2
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt

def mask_image(image, mask_info):
    """
    根据 mask_info 掩膜图像，掩膜部分置为0。
    
    参数:
        image (np.ndarray): 输入图像（2D 或 3D）。
        mask_info (dict): 掩膜信息，格式为 {'top': X, 'bottom': Y, 'left': A, 'right': B}。
    
    返回:
        masked_image (np.ndarray): 掩膜后的图像（原始尺寸）。
    """
    # 检查输入合法性
    if not isinstance(mask_info, dict) or any(k not in mask_info for k in ['top', 'bottom', 'left', 'right']):
        raise ValueError("padding_info 必须包含 'top', 'bottom', 'left', 'right' 键！")
    
    # 计算裁剪区域
    h, w = image.shape[:2]
    top, bottom = int(mask_info['top']), int(mask_info['bottom'])
    left, right = int(mask_info['left']), int(mask_info['right'])
    
    # 确保裁剪范围有效
    if top + bottom >= h or left + right >= w:
        raise ValueError("掩膜区域超过图像尺寸，无法掩膜！")
    
    # 执行裁剪（支持2D和3D图像）
    masked_image = np.zeros_like(image, dtype=image.dtype)
    if len(image.shape) == 2:
        masked_image[top : h - bottom, left : w - right] = image[top : h - bottom, left : w - right]
    else:  # 3D图像（如 RGB）
        masked_image[top : h - bottom, left : w - right, :] = image[top : h - bottom, left : w - right, :]
    
    return masked_image

input = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use.tif'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use_masked.tif'
top = 0
bottom = 36
left = 0
right = 36

rgb = 1
mask_info = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
if __name__ == "__main__":
    gt = Gdal_Tool(input)
    img = gt.read_tif_to_image(rgb, to_int=False)
    img = mask_image(img, mask_info=mask_info) # 掩膜边界
    img = img.astype(np.float32)
    plt.imsave(out_path[:-4]+'.png', img , cmap='gray')
    gt.save_tif(out_path, img)