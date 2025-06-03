'''超像素分割+随机采样'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np

input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat"
out_shp = 'out.shp'

enhance_func = 'PCA' # 可选'MNF'
embedding_nums = 10 # 降维维度
samples = 4000 # 采样数量
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif)  # 使用原始数据的增强影像
    print(f'原始像素数量：{img.rows * img.cols}')

    res = img.superpixel_sampling(n_segments=1024, compactness=25, niters=2000, samples=samples, embedding_nums=embedding_nums,
                                  f='PCA', threshold=0)
    print(f'采样数量：{np.sum(res)}')
    img.create_vector(res, out_shp) # 创建shp文件