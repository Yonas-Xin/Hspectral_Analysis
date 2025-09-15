'''超像素分割+随机采样'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
import matplotlib.pyplot as plt

input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat"
out_shp = r'c:\Users\85002\Desktop\TempDIR\out.shp'

n_segments = 512 # 调整超像素数量
enhance_func = 'MNF' # 可选'MNF'
embedding_nums = 12 # 降维维度
max_samples = 20
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif, init_fig=True)  # 使用原始数据的增强影像
    print(f'原始像素数量：{img.rows * img.cols}')
    slic_label, slic_img = img.slic(n_segments=n_segments, compactness=25, n_components=embedding_nums)
    plt.imshow(slic_img)
    plt.show()

    if enhance_func == 'MNF':
        img.image_enhance(f=enhance_func, n_components=embedding_nums)
    res = img.superpixel_sampling(slic_label, img.enhance_data, max_samples=max_samples, niters=2000, threshold=0, centered=False)
    img.create_vector(res, out_shp) # 创建单个shp文件，二维矩阵转点shp文件