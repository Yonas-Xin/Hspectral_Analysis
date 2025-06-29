from Models.Encoder import *
from Models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from Models.Data import Dataset_3D, Dataset_1D
class Constrastive_learning_Model(nn.Module):
    def __init__(self, out_embedding=24, out_classes=8, in_shape=(138,17,17)):
        super().__init__()
        self.encoder = Spe_Spa_Attenres_Encoder(out_embedding=out_embedding, in_shape=in_shape) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=4096)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    def freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_from_contrastive_model(self, pth, map_location=None):
        whole_model = torch.load(pth, map_location=map_location)['model']  # 加载文件的模型部分,包括解码器权重
        encoder_state_dict = {k.replace('encoder.',''): v for k, v in whole_model.items() if "encoder." in k}  # 提取encoder权重
        self.encoder.load_state_dict(encoder_state_dict, strict=True)  # 匹配键值
    
    def gradually_unfreeze_encoder_modules(self, epoch):
        """
        根据 当前epoch逐步 解冻 encoder 的模块
        epoch:当前训练的轮次
        """
        schedule = self.encoder.get_unfreeze_plan # 获取解冻字典
        if epoch in schedule:
            modules_to_unfreeze = schedule[epoch]
            if isinstance(modules_to_unfreeze, str):
                modules_to_unfreeze = [modules_to_unfreeze]
            for name in modules_to_unfreeze:
                module = getattr(self.encoder, name)
                for param in module.parameters():
                    param.requires_grad = True
                print(f"[Epoch {epoch}] 解冻 encoder 模块: {name}")

class Shallow_3DCNN(nn.Module):
    '''浅层3D CNN模型'''
    def __init__(self, out_embedding=None, out_classes=8, in_shape=(138,17,17)):
        super().__init__()
        self.encoder = Shallow_3DCNN_Encoder(in_shape=in_shape) # 3d卷积残差编码器
        self.decoder = nn.Linear(128, out_features=out_classes)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Shallow_1DCNN(nn.Module):
    '''浅层1D CNN模型'''
    def __init__(self, out_embedding=None, out_classes=8, in_shape=(138,)):
        super().__init__()
        self.encoder = Shallow_1DCNN_Encoder(in_shape=in_shape)  # 1D CNN 编码器
        self.decoder = deep_classfier(128, out_classes, mid_channels=4096)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 3:
            raise ValueError(f"Expected input dimension 2 or 3, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
MODEL_DICT = {
    'SRACN':Constrastive_learning_Model,
    'Shallow_1DCNN':Shallow_1DCNN,
    'Shallow_3DCNN':Shallow_3DCNN
}
DATASET_DICT = {
    'SRACN':Dataset_3D,
    'Shallow_1DCNN':Dataset_1D,
    'Shallow_3DCNN':Dataset_3D
}