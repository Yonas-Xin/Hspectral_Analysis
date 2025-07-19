"""将高光谱影裁剪为多个块，针对每个块使用SMACC"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image, mnf_standard
import numpy as np
import spectral as spy
import matplotlib.pyplot as plt
from algorithms import smacc, smacc_gpu, noise_estimation
input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat"
out_shp = 'out.shp'
row, col = 3, 3
enhance_func = 'MNF' # 可选'MNF'
embedding_nums = 12 # 降维维度
samples = 4000 # 采样数量
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif, init_fig=False)
    rows, cols = img.rows, img.cols
    if rows % row == 0: row_split = rows // row
    else: row_split = rows // row + 1
    if cols % col == 0: col_split = cols // col
    else: col_split = cols // col + 1
    full_mask = np.zeros((rows, cols), dtype=np.int16)
    for i, input in enumerate(img.block_generator((row_split, col_split))): # 将影像裁剪为多个块，分解进行smacc
        bands, H, W = input.shape
        input = input.transpose(1,2,0)
        noise = noise_estimation(input)
        input = mnf_standard(input, noise, 24)
        S, F, R, mask = smacc_gpu(input, 500)

        block_row = i // col
        block_col = i % col
        start_row = block_row * row_split
        start_col = block_col * col_split
        full_mask[start_row:start_row+H, start_col:start_col+W] = mask # 结果合并，结果是一个二维掩膜，1值代表选中的端元
    img.create_vector(full_mask, r'c:\Users\85002\Desktop\TempDIR\gxt\out.shp')