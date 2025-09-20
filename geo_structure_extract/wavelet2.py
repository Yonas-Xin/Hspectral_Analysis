import pywt
import numpy as np
from skimage.filters import threshold_otsu
from image_stretch import Gdal_Tool
import matplotlib.pyplot as plt
def pad_image_for_swt2(image, level, pad_value=0):
    """
    对输入图像进行边缘填充，使其尺寸能被 2^level 整除，以满足 pywt.swt2 的要求。
    
    参数:
        image (np.ndarray): 输入图像（2D 或 3D，如 H×W 或 H×W×C）。
        level (int): SWT 分解层数，要求图像尺寸能被 2^level 整除。
        pad_value (int/float): 填充值，默认为 0。
    
    返回:
        padded_image (np.ndarray): 填充后的图像，尺寸满足 2^level 整除。
        padding_info (dict): 填充信息，格式为 {'top': X, 'bottom': Y, 'left': A, 'right': B}。
    """
    if len(image.shape) not in (2, 3):
        raise ValueError("输入图像必须是 2D (H×W) 或 3D (H×W×C) 格式！")
    
    # 计算目标尺寸（能被 2^level 整除）
    div = 2 ** level
    h, w = image.shape[:2]
    
    # 计算需要填充的像素数
    pad_h = (div - (h % div)) % div
    pad_w = (div - (w % div)) % div
    
    # 分配填充区域（均匀分布在上下/左右）
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # 对 2D 灰度图或 3D 彩色图进行填充
    if len(image.shape) == 2:
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode='constant',
            constant_values=pad_value
        )
    else:  # 3D 图像（如 RGB）
        padded_image = np.pad(
            image,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=pad_value
        )
    
    # 返回填充后的图像和填充信息
    padding_info = {
        'top': pad_top,
        'bottom': pad_bottom,
        'left': pad_left,
        'right': pad_right
    }
    return padded_image, padding_info

def crop_image(padded_image, padding_info, factor=1):
    """
    根据 padding_info 裁剪图像，移除填充部分，恢复原始尺寸。
    
    参数:
        padded_image (np.ndarray): 填充后的图像（2D 或 3D）。
        padding_info (dict): 填充信息，格式为 {'top': X, 'bottom': Y, 'left': A, 'right': B}。
    
    返回:
        cropped_image (np.ndarray): 裁剪后的图像（原始尺寸）。
    """
    # 检查输入合法性
    if not isinstance(padding_info, dict) or any(k not in padding_info for k in ['top', 'bottom', 'left', 'right']):
        raise ValueError("padding_info 必须包含 'top', 'bottom', 'left', 'right' 键！")
    
    # 计算裁剪区域
    h, w = padded_image.shape[:2]
    top, bottom = int(factor*padding_info['top']), int(factor*padding_info['bottom'])
    left, right = int(factor*padding_info['left']), int(factor*padding_info['right'])
    
    # 确保裁剪范围有效
    if top + bottom >= h or left + right >= w:
        raise ValueError("填充区域超过图像尺寸，无法裁剪！")
    
    # 执行裁剪（支持2D和3D图像）
    if len(padded_image.shape) == 2:
        cropped_image = padded_image[top : h - bottom, left : w - right]
    else:  # 3D图像（如 RGB）
        cropped_image = padded_image[top : h - bottom, left : w - right, :]
    
    return cropped_image

def swt2_multiscale_edge(img, wavelet='haar', level=3, use_modulus_maxima=False, if_fusion=False):
    """
    基于SWT2的多层边缘检测
    img: 输入灰度图（numpy array）
    wavelet: 小波基
    level: 分解层数
    use_modulus_maxima: 是否启用模极大值检测
    """
    print(f"原始影像数据形状：{img.shape}")
    img = img.astype(np.float32)
    img, info = pad_image_for_swt2(img, level=level)
    print(info)
    
    # 多层分解
    coeffs = pywt.swt2(img, wavelet=wavelet, level=level)
    
    if if_fusion:
        # 用平方和累积所有尺度的高频信息
        edge_sum = np.zeros_like(img, dtype=np.float32)
        for cA, (cH, cV, cD) in coeffs:
            if use_modulus_maxima:
                # 模极大值检测
                M = np.sqrt(cH**2 + cV**2)  # 模
                theta = np.arctan2(cV, cH)  # 方向
                M_suppressed = non_max_suppression(M, theta)
                edge_sum += M_suppressed**2
            else:
                edge_sum += cH**2 + cV**2 + cD**2
    else:
        # 直接提取最大尺度的系数（coeffs[0]）
        _, (cH_max, cV_max, cD_max) = coeffs[0]  # 最大尺度的高频分量

        if use_modulus_maxima:
            # 模极大值检测（仅对最大尺度）
            M = np.sqrt(cH_max**2 + cV_max**2)  # 模
            theta = np.arctan2(cV_max, cH_max)  # 方向
            edge_sum = non_max_suppression(M, theta)  # 非极大值抑制
        else:
            # 直接合并最大尺度的高频信息（H+V+D）
            edge_sum = np.sqrt(cH_max**2 + cV_max**2 + cD_max**2)

    # 得到增强边缘图
    edges = np.sqrt(edge_sum)
    edges = crop_image(edges, info, factor=1)
    edges_norm = (edges - edges.min()) / (edges.max() - edges.min())
    print(f"边缘增强影像数据形状：{edges_norm.shape}")

    # Otsu 阈值分割
    thresh = threshold_otsu(edges)
    binary = (edges >= thresh).astype(np.uint8)

    return edges_norm, binary

def non_max_suppression(M, theta):
    """
    模极大值的方向非极大值抑制
    M: 模
    theta: 梯度方向（弧度）
    """
    # 转换为角度 0~180
    angle = np.rad2deg(theta) % 180

    Z = np.zeros_like(M, dtype=np.float32)
    rows, cols = M.shape

    for i in range(1, rows-1):
        for j in range(1, cols-1):
            try:
                q = 255
                r = 255

                # 水平
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = M[i, j+1]
                    r = M[i, j-1]
                # -45°方向
                elif (22.5 <= angle[i, j] < 67.5):
                    q = M[i+1, j-1]
                    r = M[i-1, j+1]
                # 垂直方向
                elif (67.5 <= angle[i, j] < 112.5):
                    q = M[i+1, j]
                    r = M[i-1, j]
                # 45°方向
                elif (112.5 <= angle[i, j] < 157.5):
                    q = M[i-1, j-1]
                    r = M[i+1, j+1]

                if (M[i, j] >= q) and (M[i, j] >= r):
                    Z[i, j] = M[i, j]
                else:
                    Z[i, j] = 0

            except IndexError:
                pass

    return Z

input_tif = r'C:\Users\85002\Desktop\TempDIR\out.dat'
out_path = r'c:\Users\85002\Desktop\TempDIR\test2\binary_use.tif'
level = 5

stretch = "Linear_2%" # Linear_2% or Linear
rgb = (1,2,3) # rgb 组合，从1开始。(1,2,3) or 1
if '__main__' == __name__:
    gt = Gdal_Tool(input_tif)
    image = gt.read_tif_to_image(rgb, stretch=stretch)
    edges_norm, binary = swt2_multiscale_edge(image, wavelet='bior2.2', level=level, use_modulus_maxima=False)

    plt.imsave(out_path[0:-4]+'_wavelet.png', edges_norm , cmap='gray')
    plt.imsave(out_path[0:-4]+'_binary.png', binary , cmap='gray')
    gt.save_tif(out_path, binary)