import sys, os  
sys.path.append('.')
from core import Hyperspectral_Image, mnf_standard
import numpy as np
from algorithms import smacc_gpu, noise_estimation
import argparse

def smacc_sampling(input_tif, output_dir, row=3, col=3, embedding_nums=12, samples=4000):
    """
    将高光谱影裁剪为多个块，针对每个块使用SMACC
    :param input_tif: 输入影像路径
    :param output_dir: 输出shp路径
    :param row: 行分块数
    :param col: 列分块数
    :param embedding_nums: 降维维度
    :param samples: 采样数量
    """
    try:
        img = Hyperspectral_Image()
        img.init(input_tif, init_fig=False)
        rows, cols = img.rows, img.cols
        if rows % row == 0: row_split = rows // row
        else: row_split = rows // row + 1
        if cols % col == 0: col_split = cols // col
        else: col_split = cols // col + 1
        full_mask = np.zeros((rows, cols), dtype=np.int16)
        for i, input in enumerate(img.block_generator((row_split, col_split))): # 将影像裁剪为多个块，分解进行smacc
            bands, H, W = input.shape
            input = input.transpose(1,2,0)
            noise = noise_estimation(input)
            input = mnf_standard(input, noise, embedding_nums)
            single_samples = int(samples*(H*W)/(rows*cols))
            S, F, R, mask = smacc_gpu(input, single_samples)

            block_row = i // col
            block_col = i % col
            start_row = block_row * row_split
            start_col = block_col * col_split
            full_mask[start_row:start_row+H, start_col:start_col+W] = mask # 结果合并，结果是一个二维掩膜，1值代表选中的端元
        img.create_vector(full_mask, output_dir)
        return True, "采样成功"
    except Exception as e:
        return False, f"采样失败: {e}"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_tif", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--row", type=int, default=3)
    parser.add_argument("--col", type=int, default=3)
    parser.add_argument("--embedding_nums", type=int, default=12)
    parser.add_argument("--samples", type=int, default=4000)
    args = parser.parse_args()

    smacc_sampling(
        input_tif=os.path.abspath(args.input_tif),
        output_dir=os.path.abspath(args.output_dir),
        row=args.row,
        col=args.col,
        embedding_nums=args.embedding_nums,
        samples=args.samples
    )