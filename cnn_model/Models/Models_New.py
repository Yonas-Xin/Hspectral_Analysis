"""针对Models模块的升级"""
from cnn_model.Models.Encoder import *
from cnn_model.Models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from cnn_model.Models.Data import Dataset_3D, Dataset_1D

class My_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfreeze_idx = None

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.fc.parameters():
            param.requires_grad = True # encoder 的fc层需要正常梯度传播
        
        for param in self.encoder.layer4.parameters():
             param.requires_grad = True

    def _load_encoer_params(self, state_dict):
        try:
            self.encoder.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            # 过滤掉不匹配的键
            print('Atteneion: The encoder weights do not match exactly!')
            model_state_dict = self.encoder.state_dict()
            matched_state_dict = {
                k: v for k, v in state_dict.items() 
                if k in model_state_dict and v.shape == model_state_dict[k].shape
            }
            model_state_dict.update(matched_state_dict)
            self.encoder.load_state_dict(model_state_dict, strict=False)
            skipped = set(state_dict.keys()) - set(matched_state_dict.keys())
            if skipped:
                print(f"Skipped loading these keys due to size mismatch: {skipped}")

class Res_3D_18Net(My_Model):
    def __init__(self, out_classes, out_embeddings=1024):
        super().__init__()
        self.encoder = ResNet_3D(block=Basic_Residual_block, layers=[2,2,2,2], num_classes=out_embeddings) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embeddings, out_classes, mid_channels=4096)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Res_3D_34Net(My_Model):
    def __init__(self, out_classes, out_embeddings=1024):
        super().__init__()
        self.encoder = ResNet_3D(block=Basic_Residual_block, layers=[3,4,6,3], num_classes=out_embeddings) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embeddings, out_classes, mid_channels=4096)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Res_3D_50Net(My_Model):
    def __init__(self, out_classes, out_embeddings=1024):
        super().__init__()
        self.encoder = ResNet_3D(block=Bottleneck_Residual_block, layers=[3,4,6,3], num_classes=out_embeddings) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embeddings, out_classes, mid_channels=4096)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

MODEL_DICT = {
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net
}
DATASET_DICT = {
    "Res_3D_18Net": Dataset_3D,
    "Res_3D_34Net": Dataset_3D,
    "Res_3D_50Net": Dataset_3D
}