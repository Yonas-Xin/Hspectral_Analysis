import pywt
import numpy as np
from skimage.filters import threshold_otsu
from image_stretch import Gdal_Tool
# import matplotlib.pyplot as plt
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

def swt2_multiscale_edge(img, wavelet='haar', level=3, use_modulus_maxima=False):
    """
    基于SWT2的多层边缘检测
    img: 输入灰度图（numpy array）
    wavelet: 小波基
    level: 分解层数
    use_modulus_maxima: 是否启用模极大值检测
    """
    img = img.astype(np.float32)
    img, info = pad_image_for_swt2(img, level=level)
    
    # 多层分解
    coeffs = pywt.swt2(img, wavelet=wavelet, level=level)
    
    # 直接提取最大尺度的系数（coeffs[0]）
    _, (cH_max, cV_max, cD_max) = coeffs[0]  # 最大尺度的高频分量

    if use_modulus_maxima:
        # 模极大值检测（仅对最大尺度）
        M = np.sqrt(cH_max**2 + cV_max**2)  # 模
        theta = np.arctan2(cV_max, cH_max)  # 方向
        edges = calculate_modulus_maxima(M, theta)  # 非极大值抑制
    else:
        # 直接合并最大尺度的高频信息（H+V+D）
        edges = np.sqrt(cH_max**2 + cV_max**2)

    edges = crop_image(edges, info, factor=1)
    # edges_norm = (edges - edges.min()) / (edges.max() - edges.min()) # 拉伸显示边缘增强图像

    # Otsu 阈值分割
    thresh = threshold_otsu(edges)
    binary = (edges >= thresh).astype(np.uint8)

    return binary

def calculate_modulus_maxima(modulus, angle):
    """
    code form: https://github.com/tdextrous/edge-detection-wavelets/blob/master/wtmm.m
    Compute modulus maxima for a given modulus and angle matrix.
    :param modulus: 2D numpy array
    :param angle: 2D numpy array of the same shape as modulus
    :return: 2D numpy array of modulus maxima
    """
    sz1, sz2 = modulus.shape
    modulus_maxima = np.zeros((sz1, sz2))
    pi = np.pi

    for i in range(sz2):
        for j in range(sz1):
            curr_mod = modulus[j, i]
            curr_angle = angle[j, i]

            # Initialize neighbor indices
            l_neighbor_index = [None, None]
            r_neighbor_index = [None, None]

            # Determine modulus neighbors indices along angle
            if (0 <= curr_angle < (pi/8)) or \
               ((7*pi/8) <= curr_angle < (9*pi/8)) or \
               ((15*pi/8) <= curr_angle < (2*pi)):
                l_neighbor_index = [j, i-1]
                r_neighbor_index = [j, i+1]

            elif ((pi/8) <= curr_angle < (3*pi/8)) or \
                 ((9*pi/8) <= curr_angle < (11*pi/8)):
                l_neighbor_index = [j+1, i-1]
                r_neighbor_index = [j-1, i+1]

            elif ((3*pi/8) <= curr_angle < (5*pi/8)) or \
                 ((11*pi/8) <= curr_angle < (13*pi/8)):
                l_neighbor_index = [j+1, i]
                r_neighbor_index = [j-1, i]

            elif ((5*pi/8) <= curr_angle < (7*pi/8)) or \
                 ((13*pi/8) <= curr_angle < (15*pi/8)):
                l_neighbor_index = [j-1, i-1]
                r_neighbor_index = [j+1, i+1]

            # If indices are nondegenerate, compare against curr_mod
            is_maxima = True

            # Check left neighbor
            if l_neighbor_index[0] is not None and l_neighbor_index[1] is not None:
                if (0 <= l_neighbor_index[0] < sz1) and (0 <= l_neighbor_index[1] < sz2):
                    l_neighbor = modulus[l_neighbor_index[0], l_neighbor_index[1]]
                    if l_neighbor > curr_mod:
                        is_maxima = False

            # Check right neighbor
            if r_neighbor_index[0] is not None and r_neighbor_index[1] is not None:
                if (0 <= r_neighbor_index[0] < sz1) and (0 <= r_neighbor_index[1] < sz2):
                    r_neighbor = modulus[r_neighbor_index[0], r_neighbor_index[1]]
                    if r_neighbor > curr_mod:
                        is_maxima = False

            # Add value if good
            if is_maxima:
                modulus_maxima[j, i] = curr_mod

    return modulus_maxima


input_tif = r'C:\Users\85002\Desktop\TempDIR\out2.dat'
out_path = r'c:\Users\85002\Desktop\TempDIR\wavelet5.tif'
level = 5
use_modulus_maxima = True # 是否启用模极大值检测

wavelet = 'haar' # haar, db1, db2, sym2, coif1，bior2.2
stretch = "Linear_2%" # Linear_2% or Linear
rgb = (1,2,3) # rgb 组合，从1开始。(1,2,3) or 1
if '__main__' == __name__:
    gt = Gdal_Tool(input_tif)
    image = gt.read_tif_to_image(rgb, stretch=stretch)
    binary = swt2_multiscale_edge(image, wavelet='haar', level=level, use_modulus_maxima=use_modulus_maxima)

    # plt.imsave(out_path[0:-4]+'_binary.png', binary , cmap='gray')
    gt.save_tif(out_path, binary)