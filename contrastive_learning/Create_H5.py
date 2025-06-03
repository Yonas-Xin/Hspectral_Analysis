import os
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent  # 根据实际情况调整
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import h5py
from contrastive_learning.Models.Data import SSF_3D,read_tif_with_gdal
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import read_txt_to_list
def create_h5_dataset(tif_paths, output_h5_path, batch=32):
    """
    创建HDF5数据集，根据path将所有tif数据压入h5格式的数据中，data为数据集
    数据形状为（num, C, H, W）
    :param tif_paths: TIF文件路径列表
    :param output_h5_path: 输出的HDF5文件路径
    :param chunk_size: HDF5 chunk大小
    """
    datasets = SSF_3D(tif_paths)
    dataloader = DataLoader(datasets, batch_size=batch, shuffle=False, num_workers=4)
    sample_shape = read_tif_with_gdal(tif_paths[0].split()[0]).shape # 获取第一个样本的形状 (138, 25, 25)
    num_samples = len(datasets)
    with h5py.File(output_h5_path, 'w') as hf:
        # 创建可扩展的数据集
        data_dset = hf.create_dataset('data',
                                      shape=(num_samples, *sample_shape),
                                      # maxshape=(None, *sample_shape),  # 允许后续扩展
                                      dtype='float32',
                                      chunks=(1, *sample_shape),
                                      # compression='gzip'
                                      )
        # 逐步填充数据
        pos = 0  # 记录数据存入的索引
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.numpy()
            batch = data.shape[0]
            data_dset[pos:pos+batch] = data
            pos += batch

if __name__ == '__main__':
    output_h5_name = 'contrastive_learning_138_17_17_1.h5'
    output_h5_path = os.path.join(os.getcwd(),output_h5_name)
    path1 = read_txt_to_list(r'D:\Programing\pythonProject\Hyperspectral_Analysis\block_clip_for_contrastive_learning1\.datasets.txt')
    path2 = read_txt_to_list(r'D:\Programing\pythonProject\Hyperspectral_Analysis\block_clip_for_contrastive_learning2\.datasets.txt')
    path = path1+path2
    create_h5_dataset(path, output_h5_name, 128)
    pass
