import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from gdal_utils import batch_random_split_point_shp

if __name__ == '__main__':
    input_shp_dir = r'/'
    output_shp_dir = r'/'
    num_to_select = 100 # 要随机选取的要素数量, 如果小于1将按照比例选取
    batch_random_split_point_shp(input_shp_dir, output_shp_dir, num_to_select)