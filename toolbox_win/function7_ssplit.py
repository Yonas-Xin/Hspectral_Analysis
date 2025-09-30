import sys, os
sys.path.append('.')
from gdal_utils import batch_random_split_point_shp
import argparse

def sample_split(input_shp_dir, output_shp_dir, num_to_select):
    try:
        batch_random_split_point_shp(input_shp_dir, output_shp_dir, num_to_select)
    except Exception as e:
        return False, f"样本划分失败: {e}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_shp_dir", type=str, required=True, help="输入 shp 文件夹路径")
    parser.add_argument("--output_shp_dir", type=str, required=True, help="输出 shp 文件夹路径")
    parser.add_argument("--num_to_select", type=float, default=0.6, help="要随机选取的要素数量, 如果小于1将按照比例选取")
    args = parser.parse_args()
    sample_split(
        input_shp_dir=os.path.abspath(args.input_shp_dir),
        output_shp_dir=os.path.abspath(args.output_shp_dir),
        num_to_select=args.num_to_select
    )