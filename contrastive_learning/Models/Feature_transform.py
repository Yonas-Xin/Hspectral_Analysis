import torch
import torch.nn as nn
import kornia.augmentation as K
from typing import Tuple
import random

class RandomSpectralMask(nn.Module):
    """随机掩膜每个像元光谱波段"""
    def __init__(self, mask_prob: float = 0.5, p: float = 0.5):
        super().__init__()
        self.mask_prob = mask_prob
        self.p = p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.mask_prob == 0:
            return x
            
        # 为每个空间位置(H,W)生成独立的随机掩膜
        B, C, H, W = x.shape
        mask = torch.rand(B, C, H, W, device=x.device) < self.p
        batch_mask = torch.rand(B, 1, 1, 1, device=x.device) < self.mask_prob
        batch_mask = batch_mask.expand(B, C, H, W)
        mask = (mask | batch_mask).float()
        return x * mask

class BandDropout(nn.Module):
    def __init__(self, p=0.2, drop_prob=0.5):
        super().__init__()
        # self.dropout = nn.Dropout3d(p)  # 3D Dropout
        self.drop_prob = drop_prob
        self.p = p

    def forward(self, x):
        """
        输入: (batch, bands, H, W)
        输出: (batch, bands, H, W)，部分波段被随机置零
        """
        if not self.training:
            return x
        batch_size = x.shape[0]
        drop_mask = torch.rand(batch_size, 1, 1, 1, device=x.device) < self.drop_prob
        band_mask = (torch.rand(x.shape[1], 1, 1, device=x.device) > self.p).float()
        x = x * (drop_mask * band_mask + (~drop_mask))
        return x

class HighDimBatchAugment(nn.Module):
    """高维图像块（如高光谱[B,C,H,W]）的批量增强"""
    def __init__(
            self,
            crop_size: Tuple[int, int],
            spectral_mask_prob: float,
            band_dropout_prob:float,
            flip_prob: float = 0.5,
            rotate_prob: float = 0.5,
            add_gaussian_prob: float=0.5,
            erase_prob: float = 0.5,
            rotate_degrees: float = 90.0,
            crop_scale: Tuple[float, float] = (0.8, 1.0),
            crop_ratio: Tuple[float, float] = (0.9, 1.1),
            noise_std: float = 0.01,
            erase_scale: Tuple[float, float] = (0.01, 0.3),
            erase_ratio: Tuple[float, float] = (0.4, 2.5),
            spectral_mask_p: float = 0.75,
            bands_dropout_p: float = 0.2,
    ):
        super().__init__()
        # 初始化增强操作
        self.flip = K.RandomHorizontalFlip(p=flip_prob)
        self.rotate = K.RandomRotation(degrees=rotate_degrees, p=rotate_prob)
        self.crop = K.RandomResizedCrop(
            size=crop_size,
            scale=crop_scale,
            ratio=crop_ratio,
            resample='bilinear'
        )
        self.add_gaussian = K.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=add_gaussian_prob, same_on_batch=False
        )

        self.erase = K.RandomErasing(
            p=erase_prob, scale=erase_scale, ratio=erase_ratio, value=0
        )
        if spectral_mask_prob > 0:
            self.spectral_mask = RandomSpectralMask(mask_prob=spectral_mask_prob, p=spectral_mask_p)
        else: self.spectral_mask = None
        if band_dropout_prob > 0:
            self.band_dropout = BandDropout(p = bands_dropout_p, drop_prob=band_dropout_prob)
        else: self.band_dropout = None
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入: [B, C, H, W]
        输出: [B, C, crop_H, crop_W]
        """
        # 确保输入是4D张量
        if x.dim() == 3:
            x = x.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
        elif x.dim() == 5:
            x = x.squeeze(1)
        else: pass
        # （所有操作自动支持批量）
        x = self.flip(x, inplace=True)  # 随机水平翻转
        x = self.rotate(x)  # 随机旋转
        x = self.crop(x)  # 随机裁剪
        x = self.add_gaussian(x) # 随机添加高斯噪声
        x = self.erase(x) # 随机擦除
        if self.spectral_mask is not None:
            x = self.spectral_mask(x) # 光谱随机掩膜\
        if self.band_dropout is not None:
            x = self.band_dropout(x) # 随机丢弃波段
        return x

if __name__ == "__main__":
    # 测试特征转换模块的功能
    from osgeo import gdal
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    import torch
    import matplotlib.pyplot as plt
    gdal.UseExceptions()
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    def read_tif_with_gdal(tif_path):
        '''读取栅格原始数据
        返回dataset[bands,H,W]'''
        dataset = gdal.Open(tif_path)
        dataset = dataset.ReadAsArray()
        if dataset.dtype == np.int16:
            dataset = dataset.astype(np.float32) * 1e-4
        return dataset
    class Dataset_3D(Dataset):
        '''输入一个list文件，list元素代表数据地址'''
        def __init__(self, data_list, transform=None):
            """
            将列表划分为数据集,[batch, 1, H, w, bands]
            """
            self.image_paths = data_list
            image,_ = self.__getitem__(0)
            self.data_shape = image.shape

        def __len__(self):
            return len(self.image_paths)
        def __getitem__(self, idx):
            """
            根据索引返回图像及其标签
            image（1，C，H，W）
            """
            image_path, label = self.image_paths[idx].split()
            label = np.array(label, dtype=np.int16)
            image = read_tif_with_gdal(image_path)
            label = torch.tensor(label, dtype=torch.long)
            image = torch.tensor(image, dtype=torch.float32)
            return image, label
    
    def read_txt_to_list(filename):
        with open(filename, 'r') as file:
            # 逐行读取文件并去除末尾的换行符
            data = [line.strip() for line in file.readlines()]
        return data
    def read_dataset_from_txt(txt_file):
        'txt文件绝对地址'
        parent_dir = os.path.dirname(txt_file)
        paths = read_txt_to_list(txt_file)
        x = [os.path.basename(i) for i in paths]
        y = [os.path.join(parent_dir, i) for i in x]
        return y
    def visualize_comparison(original_tensor, augmented_tensor, band_indices=(9, 19, 29)):
        """
        Improved visualization function for hyperspectral images
        Args:
            original_tensor: Input tensor of shape [C, H, W] or [B, C, H, W]
            augmented_tensor: Augmented tensor of same shape
            band_indices: Three band indices to use for RGB visualization
        """
        def prepare_image(tensor):
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu().numpy()
            
            # Handle batch dimension
            if tensor.ndim == 4:
                tensor = tensor[0]  # Take first sample from batch
            
            # Select specified bands and transpose to [H, W, C]
            img = tensor[list(band_indices), :, :].transpose(1, 2, 0)
            
            # Normalize each band to [0,1] range
            # for band in range(img.shape[-1]):
            #     band_data = img[:, :, band]
            #     normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
            #     img[:, :, band] = normalized
                
            return img
        
        # Prepare images for comparison
        orig_img = prepare_image(original_tensor)
        aug_img = prepare_image(augmented_tensor)
        
        # Create comparison visualization
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(orig_img)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(aug_img)
        plt.title('Augmented Image')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    def plot_multiline(dataset, save_path=None):
            COLOR_TYPE0 = '#1f77b4'  # 蓝色
            
            plt.figure(figsize=(10, 6), dpi=125)
            with plt.rc_context({'axes.edgecolor': 'black',
                                'axes.linewidth': 1.5}):
                ax = plt.gca()
            
            for i, curve in enumerate(dataset):
                x = np.arange(len(curve))  # X 轴坐标（0, 1, 2, ...）
                y = curve
                # 找到所有 0 值的位置
                zero_indices = np.where(y == 0)[0]
                if i>1000:
                    break

                if len(zero_indices) == 0:
                    plt.plot(x, y, color=COLOR_TYPE0, alpha=0.8, linewidth=0.5, zorder=2, linestyle='-', label='Saved samples')
                    continue
                start_idx = 0
                for zero_idx in zero_indices:
                    # 绘制当前段（从 start_idx 到 zero_idx-1）
                    if start_idx < zero_idx:
                        plt.plot(x[start_idx:zero_idx], y[start_idx:zero_idx], color=COLOR_TYPE0, alpha=0.8, 
                                linewidth=0.5, zorder=2, linestyle='-', label='Saved samples')
                    start_idx = zero_idx + 1  # 跳过 0 值
                
                # 绘制最后一段（最后一个 0 值之后的部分）
                if start_idx < len(y):
                    plt.plot(x[start_idx:], y[start_idx:], color=COLOR_TYPE0, alpha=0.8, 
                            linewidth=0.5, zorder=2, linestyle='-', label='Saved samples')
            ax.grid(True, 
                    linestyle='--', 
                    linewidth=0.3, 
                    alpha=0.5, 
                    color='black')  # 黑色虚线网格
            
            if save_path:
                count = 1
                base_path = save_path
                while os.path.exists(save_path):
                    save_path = base_path[:-4]+str(count)+'.png'
                    count+=1
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                print(f'png file created:{save_path}')
            plt.show()

    txt_file = r"C:\Users\85002\Desktop\TempDIR\ZY-01-Test\clip_by_shpfile\.datasets.txt"
    dataset = read_dataset_from_txt(txt_file)
    dataset = Dataset_3D(dataset)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    feature_transform = HighDimBatchAugment(crop_size=(17,17), )
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    for imgs, labels in dataloader:
        B, C, H, W = imgs.shape
        imgs = imgs.to(device)
        with torch.no_grad():
            img_enhance = feature_transform.forward(imgs).cpu().numpy()
            print(img_enhance.shape)
        # visualize_comparison(imgs[0], img_enhance[0])
        # visualize_comparison(imgs[1], img_enhance[1])

        imgs = imgs[0].cpu().numpy().transpose(1,2,0).reshape(-1, C)

        img_enhance1 = img_enhance[0].transpose(1,2,0).reshape(-1, C)
        img_enhance2 = img_enhance[1].transpose(1,2,0).reshape(-1, C)
        plot_multiline(img_enhance1)
        plot_multiline(img_enhance2)
