#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from image_stretch import Gdal_Tool
from wavelet2 import swt2_multiscale_edge
from post_processing1_edge_mask import mask_image
import cv2
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.morphology import remove_small_holes
from skeletonization import skeletonize_and_prune

def swt2_edge(input_tif, out_path, level=5, wavelet='haar', stretch="Linear_2%", rgb=(1,2,3), use_modulus_maxima=True):
    """
    使用Stationary Wavelet Transform (SWT) 提取图像边缘。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - level: SWT分解的层数
    - wavelet: 使用的小波类型
    - stretch: 图像拉伸方法
    - rgb: 读取的波段组合
    - use_modulus_maxima: 是否启用模极大值检测
    """
    try:
        gt = Gdal_Tool(input_tif)
        image = gt.read_tif_to_image(rgb, stretch=stretch)
        binary = swt2_multiscale_edge(image, wavelet=wavelet, level=level, use_modulus_maxima=use_modulus_maxima)
        gt.save_tif(out_path, binary)
        return f"Edge extraction completed. Output saved to {out_path}", binary
    except Exception as e:
        return f"Error during edge extraction: {e}", None
    
def canny_edge(input_tif, out_path, th1=50, th2=100, down_sample_factor=1, down_sample_func="NEAREST", stretch="Linear", rgb=(1,2,3)):
    """
    使用Canny算子提取图像边缘。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - th1: Canny算子的第一个阈值
    - th2: Canny算子的第二个阈值
    - down_sample_factor: 降采样倍数
    - down_sample_func: 降采样方法
    - stretch: 图像拉伸方法
    - rgb: 读取的波段组合
    """
    try:
        gt = Gdal_Tool(input_tif)
        img = gt.read_tif_to_image(rgb, stretch=stretch, to_int=True, to_gray=True)
        if down_sample_factor > 1:
            img = gt.down_sample(img, factor=down_sample_factor, FUNC=down_sample_func)
        
        edges = cv2.Canny(img.astype(np.uint8), th1, th2)
        gt.save_tif(out_path, edges, factor=down_sample_factor if down_sample_factor > 1 else None)
        return f"Canny edge detection completed. Output saved to {out_path}", edges
    except Exception as e:
        return f"Error during Canny edge detection: {e}", None

def sobel_edge(input_tif, out_path, th=10, down_sample_factor=1, down_sample_func="NEAREST", stretch="Linear", rgb=(1,2,3)):
    """
    使用Sobel算子提取图像边缘。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - th: 阈值分割，用于将边缘结果二值化的阈值
    - down_sample_factor: 降采样倍数
    - down_sample_func: 降采样方法
    - stretch: 图像拉伸方法
    - rgb: 读取的波段组合
    """
    try:
        gt = Gdal_Tool(input_tif)
        img = gt.read_tif_to_image(rgb, stretch=stretch, to_int=True, to_gray=True)
        if down_sample_factor > 1:
            img = gt.down_sample(img, factor=down_sample_factor, FUNC=down_sample_func)

        x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(img, cv2.CV_16S, 0, 1)

        Scale_absX = cv2.convertScaleAbs(x)
        Scale_absY = cv2.convertScaleAbs(y)
        result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)
        result[result > th] = 255

        gt.save_tif(out_path, result, factor=down_sample_factor if down_sample_factor > 1 else None)
        return f"Sobel edge detection completed. Output saved to {out_path}", result
    except Exception as e:
        return f"Error during Sobel edge detection: {e}", None
    
def post_mask_image(input_tif, out_path, top=0, bottom=0, left=0, right=0):
    """
    对二值图像进行边缘掩膜处理。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - top: 上边缘宽度
    - bottom: 下边缘宽度
    - left: 左边缘宽度
    - right: 右边缘宽度
    """
    try:
        gt = Gdal_Tool(input_tif)
        rgb = 1
        img = gt.read_tif_to_image(rgb, to_int=False)
        mask_info = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
        img = mask_image(img, mask_info=mask_info)
        img = img.astype(np.float32)
        gt.save_tif(out_path, img)
        return f"Image masking completed. Output saved to {out_path}", img
    except Exception as e:
        return f"Error during image masking: {e}", None

def post_remove_small_obj(input_tif, out_path, area=1000000):
    """
    去除二值图像中的小物体。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - area: 小物体面积阈值，小于该值的物体将被移除
    """
    try:
        gt = Gdal_Tool(input_tif)
        rgb = 1
        img = gt.read_tif_to_image(rgb, to_int=False)
        img = img.astype(np.bool)
        img = remove_small_objects(img, area)
        img = img.astype(np.float32)
        gt.save_tif(out_path, img)
        return f"Small object removal completed. Output saved to {out_path}", img
    except Exception as e:
        return f"Error during small object removal: {e}", None

def post_remove_small_hole(input_tif, out_path, area=1000000):
    """
    填充二值图像中的小孔洞。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - area: 小孔洞面积阈值，小于该值的孔洞将被填充
    """
    try:
        gt = Gdal_Tool(input_tif)
        rgb = 1
        img = gt.read_tif_to_image(rgb, to_int=False)
        img = img.astype(np.bool)
        img = remove_small_holes(img, area_threshold=area)
        img = img.astype(np.float32)
        gt.save_tif(out_path, img)
        return f"Small hole removal completed. Output saved to {out_path}", img
    except Exception as e:
        return f"Error during small hole removal: {e}", None

def post_erode_image(input_tif, out_path, kernel_size=(5,5), iterations=1):
    """
    对二值图像进行腐蚀操作。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - kernel_size: 腐蚀核的大小
    - iterations: 腐蚀操作的迭代次数
    """
    try:
        gt = Gdal_Tool(input_tif)
        rgb = 1
        img = gt.read_tif_to_image(rgb, to_int=False)
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img = cv2.erode(img, kernel, iterations=iterations)
        img = img.astype(np.float32)
        gt.save_tif(out_path, img)
        return f"Erosion completed. Output saved to {out_path}", img
    except Exception as e:
        return f"Error during erosion: {e}", None

def post_dilate_image(input_tif, out_path, kernel_size=(5,5), iterations=1):
    """
    对二值图像进行膨胀操作。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的tif文件路径
    - kernel_size: 膨胀核的大小
    - iterations: 膨胀操作的迭代次数
    """
    try:
        gt = Gdal_Tool(input_tif)
        rgb = 1
        img = gt.read_tif_to_image(rgb, to_int=False)
        img = img.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        img = cv2.dilate(img, kernel, iterations=iterations)
        img = img.astype(np.float32)
        gt.save_tif(out_path, img)
        return f"Dilation completed. Output saved to {out_path}", img
    except Exception as e:
        return f"Error during dilation: {e}", None

def skeletonize_image(input_tif, out_path, min_branch_length = 1000000):
    """
    对二值图像进行细化操作。
    
    参数:
    - input_tif: 输入的tif文件路径
    - out_path: 输出的shp文件路径
    """
    try:
        rgb = 1
        gt = Gdal_Tool(input_tif)
        img = gt.read_tif_to_image(rgb, to_int=True)
        binary = skeletonize_and_prune(binary_img=img, min_branch_length=min_branch_length)
        gt.skeleton_to_shp_from_raster(binary, out_path)
        return f"Skeletonization completed. Output saved to {out_path}", binary
    except Exception as e:
        return f"Error during skeletonization: {e}", None