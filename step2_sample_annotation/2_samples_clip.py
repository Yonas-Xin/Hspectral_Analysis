'''
根据样本.shp文件进行样本的裁剪
'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from gdal_utils import crop_image_by_mask,write_list_to_txt,vector_to_mask
from core import Hyperspectral_Image
import numpy as np
import os
if __name__ == "__main__":
    img = Hyperspectral_Image()
    area_data = r'D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat' # 裁剪区域栅格影像
    label_mask = r"D:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\handle_samples_15classes_add_valdataset.shp" # 裁剪shp文件
    dir_name = r'd:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\clip_test_dataset_1x1' # 设置一个目录存放裁剪的数据

    img.init(area_data)
    img.sampling_position = img.create_mask(label_mask) # 指定sampling_position
    print(np.sum(img.sampling_position>0))
    img.crop_image_by_mask_block(dir_name, image_block=256, block_size=1)