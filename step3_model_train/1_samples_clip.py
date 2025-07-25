'''
根据样本.shp文件进行样本的裁剪
'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from gdal_utils import clip_by_multishp, write_list_to_txt, vector_to_mask
from core import Hyperspectral_Image
import numpy as np
import os
if __name__ == "__main__":
    input_img = r'C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat' # 裁剪区域栅格影像
    input_dir = r"c:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\dataset\handle_shapefile" # 裁剪shp文件夹
    out_dir = r'c:\Users\85002\Desktop\TempDIR\ZY-01-Test\handle_dataset_8classes_400samplesnew' # 存储目录
    dir_name = "SAMPLES" # 存储目录下新建一个目录存放数据
    block_size = 17
    out_tif_name = "img"
    clip_by_multishp(out_dir, input_img, input_dir, block_size=block_size, out_tif_name=out_tif_name)