'''
根据样本.shp文件进行样本的裁剪
'''

from gdal_utils import crop_image_by_mask,write_list_to_txt,vector_to_mask
from core import Hyperspectral_Image
import numpy as np
import os
if __name__ == "__main__":
    img = Hyperspectral_Image()
    area_data = r'C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat' # 裁剪区域栅格影像
    label_mask = r"C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\样本点数据\点集4\samples_3.shp" # 裁剪shp文件
    dir_name = r'D:\Data\Hgy\test' # 设置一个目录存放裁剪的数据

    img.init(area_data)
    img.sampling_position = img.create_mask(label_mask) # 指定sampling_position
    print(np.sum(img.sampling_position>0))
    img.crop_image_by_mask_block(dir_name, image_block=256, block_size=17)