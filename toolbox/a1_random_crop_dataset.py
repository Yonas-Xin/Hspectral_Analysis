'''
裁剪大量无标记样本以供无监督训练
'''
import sys, os
sys.path.append('.')
from core import Hyperspectral_Image
import numpy as np

image_file_path = r'C:\Users\85002\Desktop\TempDIR\test\whole_area_138.dat' # 采样影像
filepath_dir = r'c:\Users\85002\Desktop\TempDIR\test2' # 样本保存地址
patch_size = 17 # 样本大小
sample_fraction = 0.001 # 采样比例

image_block = 512 # 分块裁剪的块大小
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(image_file_path, init_fig=True) # 初始化
    '''随机采样，生成采样矩阵'''
    img.generate_sampling_mask(sample_fraction=sample_fraction) # 采样矩阵为sampling_position
    print(f'采样数量为：{np.sum(img.sampling_position)}')

    '''中间可以将采样位置转化为点shp文件'''
    img.create_vector(img.sampling_position, os.path.join(filepath_dir, '.position.shp'))

    '''裁剪样本，filepath：指定文件夹， image_block：分块裁剪， block_size：裁剪图像块大小， scale：缩放比例'''
    img.crop_image_by_mask_block(filepath=filepath_dir, image_block=image_block, patch_size=patch_size)