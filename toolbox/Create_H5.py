
"""
将样本集压缩为h5格式
"""

import sys, os
sys.path.append('.')
from cnn_model.Models.Data import CNN_Dataset,read_tif_with_gdal
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import read_txt_to_list
import h5py

def create_h5_dataset(tif_paths, output_h5_path, chunk_size=32):
    """
    创建HDF5数据集，根据path将所有tif数据压入h5格式的数据中，data为数据集，label为标签，
    数据形状为（num, C, H, W）
    :param tif_paths: TIF文件路径列表
    :param labels: 对应的标签数组
    :param output_h5_path: 输出的HDF5文件路径
    :param chunk_size: HDF5 chunk大小
    """
    datasets = CNN_Dataset(tif_paths)
    dataloader = DataLoader(datasets, batch_size=chunk_size, shuffle=False, num_workers=4)
    sample_shape = read_tif_with_gdal(tif_paths[0].split()[0]).shape # 获取第一个样本的形状 (138, 25, 25)
    num_samples = len(datasets)
    with h5py.File(output_h5_path, 'w') as hf:
        # 创建可扩展的数据集
        data_dset = hf.create_dataset('data',
                                      shape=(num_samples, *sample_shape),
                                      # maxshape=(None, *sample_shape),  # 允许后续扩展
                                      dtype='float32',
                                      chunks=(chunk_size, *sample_shape),
                                      compression='gzip')
        # 创建标签数据集
        label_dset = hf.create_dataset('labels',
                                       shape=(num_samples,),
                                       # maxshape=(None,),
                                       dtype='int16',
                                       chunks=(chunk_size,),
                                       compression='gzip')

        # 逐步填充数据
        pos = 0  # 记录数据存入的索引
        for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data = data.numpy() # 删除轴
            label = label.numpy()
            batch = data.shape[0]
            data_dset[pos:pos+batch] = data
            label_dset[pos:pos+batch] = label
            pos += batch


if __name__ == '__main__':
    train_data_lists = read_txt_to_list('./split_dataset/train_datasets.txt')
    eval_data_lists = read_txt_to_list('./split_dataset/eval_datasets.txt')

    create_h5_dataset(train_data_lists, './train_datasets.h5', 128)
    create_h5_dataset(eval_data_lists, './eval_datasets.h5',128)