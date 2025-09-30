import sys, os
sys.path.append('.')
from gdal_utils import clip_by_multishp
import argparse

def sample_crop(input_tif, input_shp_dir, out_dir, patch_size, out_tif_name):
    try:
        clip_by_multishp(out_dir, input_tif, input_shp_dir, block_size=patch_size, out_tif_name=out_tif_name)
    except Exception as e:
        return False, f"样本裁剪失败: {e}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", type=str, required=True, help="裁剪区域栅格影像路径")
    parser.add_argument("--input_shp_dir", type=str, required=True, help="裁剪shp文件夹路径")
    parser.add_argument("--out_dir", type=str, required=True, help="存储目录路径")
    parser.add_argument("--patch_size", type=int, default=17, help="裁剪图像块大小")
    parser.add_argument("--out_tif_name", type=str, default="clip_by_shpfile", help="裁剪后影像命名")
    args = parser.parse_args()
    sample_crop(
        input_tif=os.path.abspath(args.input_tif),
        input_shp_dir=os.path.abspath(args.input_shp_dir),
        out_dir=os.path.abspath(args.out_dir),
        patch_size=args.patch_size,
        out_tif_name=args.out_tif_name
    )