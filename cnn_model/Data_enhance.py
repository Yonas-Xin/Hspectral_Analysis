import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import torch.nn as nn
import torch
from utils import read_txt_to_list,write_list_to_txt
import os.path
from cnn_model.models.Data import Moni_leaning_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import gdal_utils
from torch import Tensor
import itertools
import random

def add_gaussian_noise_torch(spectrum, std=0.02):
    """
    使用 PyTorch 给光谱数据添加高斯噪声
    """
    if not isinstance(spectrum, torch.Tensor):
        return ValueError("数据必须是Tensor类型")
    device = spectrum.device
    noise = torch.randn_like(spectrum, device=device) * std
    return spectrum + noise

class BatchAugment_3d(nn.Module):
    '''使用时最好将图像转为float类型'''
    def __init__(self,
                 flip_prob: float = 0.5,
                 rot_prob: float = 0.5,
                 gaussian_noise_std: tuple = (0.006, 0.012)):
        super().__init__()
        self.flip_prob = flip_prob
        self.rot_prob = rot_prob
        self.gaussian_noise_std = gaussian_noise_std

    def rot_90(self,x,k=1):
        return torch.rot90(x, dims=(3,4), k=k)
    def flip(self,x,dims=[3]):
        return torch.flip(x, dims=dims)
    def augment_batch(self, x: Tensor) -> Tensor:
        """
        批量数据增强, 避免循环处理
        输入形状: (batch, 1, H, W, C)
        输出形状: (batch, 1, H, W, C)
        """
        device = x.device
        batch_size,_,C,H,W = x.shape
        # 随机翻转（批量处理）
        if self.flip_prob > 0:
            h_flip_mask = torch.rand(batch_size, device=device) < self.flip_prob
            v_flip_mask = (torch.rand(batch_size, device=device) < self.flip_prob) & (~h_flip_mask)
            rot_90_mask = (~h_flip_mask & ~v_flip_mask) | (torch.rand(batch_size, device=device) < self.rot_prob)
            rot_270_mask = (torch.rand(batch_size, device=device) < self.rot_prob) & (~rot_90_mask)
            rot_180_mask = (torch.rand(batch_size, device=device) < self.rot_prob) & (~rot_90_mask) & (~rot_270_mask)
            if h_flip_mask.any():
                x[h_flip_mask] = self.flip(x[h_flip_mask], dims=[3])
            if v_flip_mask.any():
                x[v_flip_mask] = self.flip(x[v_flip_mask], dims=[4])

            if rot_90_mask.any():
                x[rot_90_mask] = self.rot_90(x[rot_90_mask], k=1)
            if rot_270_mask.any():
                x[rot_270_mask] = self.rot_90(x[rot_270_mask], k=3)
            if rot_180_mask.any():
                x[rot_180_mask] = self.rot_90(x[rot_180_mask], k=2)
        std = random.uniform(self.gaussian_noise_std[0], self.gaussian_noise_std[1]) # 随机选择噪声强度
        return add_gaussian_noise_torch(x, std)
    def forward(self, x: Tensor) -> Tensor:
        return self.augment_batch(x.clone().float())

    def generate_enhance_list(self, factor=10):
        flip_options = [[2], [3], None]
        rot_options = [1, 2, 3, None]
        gaussian_options = [0.005, 0.010, 0.015, None]
        all_combinations = list(itertools.product(flip_options, rot_options, gaussian_options))[:-1]
        self.enhance_order = random.sample(all_combinations, factor)


    def order(self, x: Tensor, idx=0): # 用来对数据进行指定形式的增强，扩展数据集。
        flip, rot, noise_std = self.enhance_order[idx]
        if flip is not None:
            x = self.flip(x, dims=flip)
        if rot is not None:
            x = self.rot_90(x, k=rot)
        if noise_std is not None:
            x = add_gaussian_noise_torch(x, std=noise_std)
        return x
def enhance_dataset(dataset_path_list, out_dir, factor=5, batch = 256):
    '''
    数据集扩充，扩充的数据形成tif文件，供检查
    :param dataset_path_list: 数据地址list
    :param factor: 扩充倍数
    :param batch: 扩充时的批量处理数
    :return: 无，创建原始数据集与增强数据集的地址和label
    '''
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        raise ValueError('文件夹已存在，为避免文件夹覆盖，请重新设置文件夹名称或者删除已存在的文件夹')

    base_name_lit = [(os.path.basename(dataset_path).split())[0] for dataset_path in dataset_path_list]
    dataset = Moni_leaning_dataset(dataset_path_list)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch, num_workers=4)
    datasets_txt = os.path.join(out_dir, '.enhance_datasets.txt')

    datasets_txt_file = open(datasets_txt,'w')
    for path in dataset_path_list:
        datasets_txt_file.write(path+'\n') # 先复制一遍原始数据集
    augment = BatchAugment_3d()
    augment.generate_enhance_list(factor=factor)
    for i in range(factor):
        current_image_pos = 0
        for data,label in tqdm(dataloader, total=len(dataloader)):
            data = data.unsqueeze(1)
            data = augment.order(data, idx = i).squeeze().numpy()
            for j in range(data.shape[0]):
                tif_name = os.path.join(out_dir,f"EH_{i}_"+base_name_lit[current_image_pos])
                datasets_txt_file.write(tif_name+f' {label[j]}\n')
                gdal_utils.write_data_to_tif(tif_name, data[j], None,None)
                current_image_pos += 1
    datasets_txt_file.flush()

# if __name__ == '__main__':
#     enhance_out_dir_name = 'enhance_data'
#     current_dir = os.getcwd()
#     enhance_out_dir = os.path.join(current_dir, enhance_out_dir_name)

#     image_paths = read_txt_to_list(r'D:\Programing\pythonProject\Hyperspectral_Analysis\research1_samples_17x17\.datasets.txt')
#     enhance_dataset(image_paths, enhance_out_dir,5 ,256) # 样本数据扩增