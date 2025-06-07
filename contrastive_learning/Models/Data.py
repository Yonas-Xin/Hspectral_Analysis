import numpy as np
import torch
from torch.utils.data import Dataset
from threading import Lock
try:
    import h5pickle
except:
    pass

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')


def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据
    返回dataset[bands,H,W]'''
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) * 1e-4
    return dataset

class SSF(Dataset):
    '''输入两个个list文件，list元素代表数据地址，一个存放窗口，一个存放光谱'''
    def __init__(self, data_list, spectral_list, transform=None):
        """
        将列表划分为数据集,[batch, C, H, W] [batch, 1, sequence]
        """
        self.image_paths = data_list
        self.spectral_lists = spectral_list

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（3，rows，cols）
        """
        image_path = self.image_paths[idx]
        spectral_path = self.spectral_lists[idx]
        image = read_tif_with_gdal(image_path)
        spectral = read_tif_with_gdal(spectral_path).transpose(1,2,0)
        spectral = np.squeeze(spectral, axis=0)

        # 转换为 PyTorch 张量
        image = torch.from_numpy(image).float()
        spectral = torch.from_numpy(spectral).float()
        return image, spectral

class SSF_3D(Dataset):
    '''输入一个list文件，list元素代表数据地址'''
    def __init__(self, data_list, transform=None):
        """
        将列表划分为数据集,[batch, 1, H, w, bands]
        """
        self.image_paths = data_list

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（3，rows，cols）
        """
        image_path = self.image_paths[idx]
        image = read_tif_with_gdal(image_path)
        image = torch.from_numpy(image).float()
        return image

class SSF_3D_H5(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.hf = h5pickle.File(self.h5_file, 'r')  # 只读模式
        self.data = self.hf['data']
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        :return: (影像数据, 标签)
        """
        img = self.data[index]  # 读取 HDF5 数据
        # img = torch.tensor(img, dtype=torch.float32)  # 转换为 Tensor
        return img
    def close(self):
        """关闭 HDF5 文件"""
        self.hf.close()