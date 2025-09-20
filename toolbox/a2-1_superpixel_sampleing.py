'''超像素分割+随机采样'''
import sys, os
sys.path.append('.')
from core import Hyperspectral_Image

input_tif = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat"
out_shp = r'c:\Users\85002\Desktop\TempDIR\out.shp'
max_samples = 30 # 控制每个超像素最大采样量
n_segments = 512 # 调整超像素数量
enhance_func = 'MNF' # 可选'MNF' "PCA" 控制ppi计算时的降维方法
embedding_nums = 12 # 控制降维维度

compactness = 25 # 超像素分割的参数，调整超像素的紧密度
ppi_niters = 2000
ppi_threshold = 0
ppi_centered = False
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif, init_fig=True)  # 使用原始数据的增强影像
    print(f'The number of pixels: {img.rows * img.cols}')
    slic_label, slic_img = img.slic(n_segments=n_segments, compactness=compactness, n_components=embedding_nums)

    if enhance_func == 'MNF':
        img.image_enhance(f=enhance_func, n_components=embedding_nums)
    res = img.superpixel_sampling(slic_label, img.enhance_data, max_samples=max_samples, 
                                  niters=ppi_niters, threshold=ppi_threshold, centered=ppi_centered)
    img.create_vector(res, out_shp) # 创建单个shp文件，二维矩阵转点shp文件