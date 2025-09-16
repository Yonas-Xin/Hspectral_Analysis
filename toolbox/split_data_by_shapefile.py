import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from gdal_utils import batch_random_split_point_shp

if __name__ == '__main__':
    input_shp_dir = r'c:\Users\85002\Desktop\TempDIR\ZY-01-Test\shapefile\classed_file\Final_sample_10classes'
    output_shp_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new'
    num_to_select = 0.6 # 要随机选取的要素数量, 如果小于1将按照比例选取
    batch_random_split_point_shp(input_shp_dir, output_shp_dir, num_to_select)