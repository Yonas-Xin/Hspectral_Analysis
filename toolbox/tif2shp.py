"""批量栅格转矢量"""
import sys, os
sys.path.append('.')
import numpy as np
from gdal_utils import batch_raster_to_vector

input_shp_dir = r'C:\Users\85002\OneDrive\文档\小论文\素材\Vgg16_net_202509131926_best.tif等8个文件\Vgg16_net_202509131926_best.tif'
output_shp_dir = r'c:\Users\85002\Desktop\TempDIR\test2'
delete_value = -1 # 需要删除的值

if __name__ == '__main__':
    batch_raster_to_vector(input_shp_dir, output_shp_dir, delete_value=delete_value)