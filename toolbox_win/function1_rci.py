import sys, os
sys.path.append('.')
from core import Hyperspectral_Image
import numpy as np
import argparse

def randeom_crop_dataset(input_tif, output_dir, patch_size=17, sample_fraction=0.001, image_block=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    try:
        img = Hyperspectral_Image()
        img.init(input_tif, init_fig=True) # 初始化
        '''随机采样，生成采样矩阵'''
        img.generate_sampling_mask(sample_fraction=sample_fraction) # 采样矩阵为sampling_position
        print(f'采样数量为：{np.sum(img.sampling_position)}')

        '''中间可以将采样位置转化为点shp文件'''
        img.create_vector(img.sampling_position, os.path.join(output_dir, '.position.shp'))

        '''裁剪样本，filepath：指定文件夹， image_block：分块裁剪， block_size：裁剪图像块大小， scale：缩放比例'''
        img.crop_image_by_mask_block(filepath=output_dir, image_block=image_block, patch_size=patch_size)
        return True, f"Successfully processed {input_tif}, samples saved in {output_dir}"
    except Exception as e:
        return False, f"Error processing {input_tif}: {e}"

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=5)
    parser.add_argument("--sample_fraction", type=float, default=0.001)
    parser.add_argument("--image_block", type=int, default=512)
    args = parser.parse_args()

    randeom_crop_dataset(
        input_tif=os.path.abspath(args.input_tif),
        output_dir=os.path.abspath(args.output_dir),
        patch_size=args.patch_size,
        sample_fraction=args.sample_fraction,
        image_block=args.image_block
    )