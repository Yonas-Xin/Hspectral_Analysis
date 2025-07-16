'''
裁剪大量无标记样本以供无监督训练
'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

image_file_path = r'D:\Data\Hgy\预处理\whole_area_138.dat' # 采样影像
filepath_dir = r'D:\Data\Hgy\test' # 样本保存地址
sample_fraction = 0.001 # 采样比例

image_block = 512 # 分块裁剪的块大小
block_size = 17 # 样本大小
scale = 1e-4 # 缩放比例
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(image_file_path, init_fig=False) # 初始化
    img.init_fig_data() # 计算背景掩膜，自动拉伸生成图像

    '''随机采样，生成采样矩阵'''
    img.generate_sampling_mask(sample_fraction=sample_fraction) # 采样矩阵为sampling_position
    print(f'采样数量为：{np.sum(img.sampling_position)}')

    '''中间可以将采样位置转化为点shp文件'''
    img.create_vector(img.sampling_position, 'out.shp')

    '''裁剪样本，filepath：指定文件夹， image_block：分块裁剪， block_size：裁剪图像块大小， scale：缩放比例'''
    img.crop_image_by_mask_block(filepath=filepath_dir, image_block=image_block, block_size=block_size, scale=scale)
    
    # 显示掩膜背景
    # plt.imshow(img.backward_mask)
    # plt.show()

    # # 显示拉伸图像
    # plt.imshow(img.ori_img)
    # plt.show()