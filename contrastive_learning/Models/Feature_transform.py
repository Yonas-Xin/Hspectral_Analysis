import torch
import torch.nn as nn
import kornia.augmentation as K
from typing import Tuple
import random

class RandomSpectralMask(nn.Module):
    """随机掩膜每个空间位置50%的光谱波段"""
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

class HighDimBatchAugment(nn.Module):
    """高维图像块（如高光谱[B,C,H,W]）的批量增强"""
    def __init__(
            self,
            crop_size: Tuple[int, int],
            flip_prob: float = 0.5,
            rotate_degrees: float = 90.0,
            crop_scale: Tuple[float, float] = (0.8, 1.0),
            crop_ratio: Tuple[float, float] = (0.9, 1.1),
            noise_std: float = 0.01,
            erase_prob: float = 0.5,
            erase_scale: Tuple[float, float] = (0.01, 0.3),
            erase_ratio: Tuple[float, float] = (0.4, 2.5),
            spectral_mask_prob: float = 0.5,
            spectral_mask_p: float = 0.5

    ):
        super().__init__()
        # 初始化增强操作
        self.flip = K.RandomHorizontalFlip(p=flip_prob)
        self.rotate = K.RandomRotation(degrees=rotate_degrees, p=0.5)
        self.crop = K.RandomResizedCrop(
            size=crop_size,
            scale=crop_scale,
            ratio=crop_ratio,
            resample='bilinear'
        )
        self.add_gaussian = K.RandomGaussianNoise(
            mean=0.0, std=noise_std, p=0.5, same_on_batch=False
        )

        self.erase = K.RandomErasing(
            p=erase_prob, scale=erase_scale, ratio=erase_ratio, value=0
        )
        self.spectral_mask = RandomSpectralMask(mask_prob=spectral_mask_prob, p=spectral_mask_p)
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
        x = self.spectral_mask(x) # 光谱随机掩膜
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
            image = self.__getitem__(0)
            self.data_shape = image.shape

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
    def read_txt_to_list(filename):
        with open(filename, 'r') as file:
            # 逐行读取文件并去除末尾的换行符
            data = [line.strip() for line in file.readlines()]
        return data
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
    def plot_multiline(dataset, curve_types=None, save_path=None, 
                    show_confidence=True):
        """
        优化后的多曲线绘图函数，支持曲线分类显示、置信区间和异常点标记
        
        参数:
            dataset: 二维numpy数组 (n_curves, n_points)
            curve_types: 一维numpy数组 (n_curves,)，值为0或1，标记曲线类型
            title: 图表标题
            save_path: 图片保存路径
            show_confidence: 是否显示置信区间
            show_points: 是否显示每个数据点
            show_outliers: 是否标记异常点
        """
        # 设置颜色（类型0和类型1分别用不同颜色）
        COLOR_TYPE0 = '#1f77b4'  # 蓝色
        COLOR_TYPE1 = '#ff7f0e'  # 橙色
        
        plt.figure(figsize=(10, 6), dpi=125)
        # 设置黑色边框
        with plt.rc_context({'axes.edgecolor': 'black',
                            'axes.linewidth': 1.5}):
            ax = plt.gca()

        # 分离保留和剔除的曲线
        keep_mask = curve_types == 1
        remove_mask = curve_types == -1
        # 计算置信区间（仅基于保留样本）
        if show_confidence and np.any(keep_mask):
            valid_data = dataset[keep_mask]
            mean = np.mean(valid_data, axis=0)
            std = np.std(valid_data, axis=0)
            upper_bound = mean + 1.96 * std
            lower_bound = mean - 1.96 * std
            
            plt.fill_between(range(dataset.shape[1]), lower_bound, upper_bound,
                            color=COLOR_TYPE0, alpha=0.2, 
                            label='95% Confidence (Kept Samples)')
        
        # 检查曲线类型输入
        if curve_types is None:
            curve_types = np.zeros(dataset.shape[0], dtype=int)
        else:
            assert len(curve_types) == dataset.shape[0], "curve_types长度必须与dataset行数一致"
            assert set(np.unique(curve_types)) <= {-1, 1}, "curve_types只能包含0和1"
        # 绘制剔除的曲线
        for i in np.where(remove_mask)[0]:
            plt.plot(dataset[i], color=COLOR_TYPE1, alpha=0.6, linewidth=0.5, zorder=1, linestyle='--', label='Deleted samples')
        # 绘制保留的曲线
        for i in np.where(keep_mask)[0]:
            plt.plot(dataset[i], color=COLOR_TYPE0, alpha=0.8, linewidth=0.5, zorder=2, linestyle='-', label='Saved samples')
        # 坐标轴和网格美化
        ax.grid(True, 
                linestyle='--', 
                linewidth=0.3, 
                alpha=0.5, 
                color='black')  # 黑色虚线网格
        # ax.xaxis.set_major_locator(MaxNLocator(integer=True))  
        # 添加图例（只显示每种类型一次）
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # 去重
        plt.legend(by_label.values(), by_label.keys(), loc='best')
        
        if save_path:
            count = 1
            base_path = save_path
            while os.path.exists(save_path):
                save_path = base_path[:-4]+str(count)+'.png'
                count+=1
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f'png file created:{save_path}')
        plt.show()

    txt_file = r"D:\Data\Hgy\research_clip_samples\.datasets.txt"
    dataset = read_txt_to_list(txt_file)
    dataset = Dataset_3D(dataset)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=2)
    feature_transform = HighDimBatchAugment(crop_size=(17,17), )
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    for imgs in dataloader:
        B, C, H, W = imgs.shape
        imgs = imgs.to(device)
        with torch.no_grad():
            img_enhance = feature_transform.forward(imgs)
        # visualize_comparison(imgs[0], img_enhance[0])
        # visualize_comparison(imgs[1], img_enhance[1])

        imgs = imgs[0].cpu().numpy().transpose(1,2,0).reshape(-1, C)
        img_enhance = img_enhance[0].cpu().numpy().transpose(1,2,0).reshape(-1, C)
        curve_type = np.ones(imgs.shape[0])
        plot_multiline(imgs, curve_types=curve_type, show_confidence=False)
        plot_multiline(img_enhance, curve_types=curve_type, show_confidence=False)
