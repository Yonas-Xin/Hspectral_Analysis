"""生成增强数据并保存为tif文件, 供机器学习使用"""
import sys, os
sys.path.append('.')
from core import Hyperspectral_Image

input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
out_tif = r'enhance_data.tif'
f = "PCA" # 'PCA' or 'MNF'
n_components = 24

if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif)
    img.image_enhance(f=f, n_components=n_components)
    img.save_tif(out_tif, img.enhance_data)