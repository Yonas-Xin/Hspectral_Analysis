import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import affine_grid,grid_sample
import torchvision.transforms.functional as TF
import kornia.augmentation as K
from typing import Tuple

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
            erase_ratio: Tuple[float, float] = (0.4, 2.5)

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
        x = self.add_gaussian(x) # 随机添加告诉噪声
        x = self.erase(x) # 随机擦除
        return x