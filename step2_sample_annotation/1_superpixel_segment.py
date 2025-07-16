'''超像素分割+随机采样'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np

input_tif = r"d:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat"
out_shp = 'out.shp'

enhance_func = 'MNF' # 可选'MNF'
embedding_nums = 12 # 降维维度
samples = 4000 # 采样数量
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif)  # 使用原始数据的增强影像
    print(f'原始像素数量：{img.rows * img.cols}')

    res = img.superpixel_sampling(n_segments=512, compactness=25, niters=2000, samples=samples, embedding_nums=embedding_nums,
                                  f=enhance_func, threshold=0)
    print(f'采样数量：{np.sum(res)}')
    img.create_vector(res, out_shp) # 创建单个shp文件