"""针对Models模块的升级"""
from cnn_model.Models.Encoder import *
from cnn_model.Models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from cnn_model.Models.Data import Dataset_3D, Dataset_1D

class My_Model(nn.Module):
    def __init__(self, out_classes=None, out_embedding=None, in_shape=None): # 框架需要三个输入
        super().__init__()
        self.unfreeze_idx = None

    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.encoder.fc.parameters():
            param.requires_grad = True # encoder 的fc层需要正常梯度传播
        


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
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = ResNet_3D(block=Basic_Residual_block, layers=[2,2,2,2], num_classes=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Res_3D_34Net(My_Model):
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = ResNet_3D(block=Basic_Residual_block, layers=[3,4,6,3], num_classes=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Res_3D_50Net(My_Model):
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = ResNet_3D(block=Bottleneck_Residual_block, layers=[3,4,6,3], num_classes=out_embedding) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Shallow_3DCNN(My_Model):
    '''浅层3D CNN模型'''
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = Shallow_3DCNN_Encoder(out_embedding=out_embedding) # 3d卷积残差编码器
        self.decoder = nn.Linear(128, out_features=out_classes)
    
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Shallow_1DCNN(My_Model):
    '''浅层1D CNN模型'''
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = Shallow_1DCNN_Encoder(out_embedding=out_embedding)  # 1D CNN 编码器
        self.decoder = deep_classfier(128, out_classes, mid_channels=1024)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 3:
            raise ValueError(f"Expected input dimension 2 or 3, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class Constrastive_learning_Model(My_Model):
    def __init__(self, out_classes, out_embedding=1024, in_shape=None):
        super().__init__()
        if in_shape is None:
            raise ValueError("in_shape must be provided for the model.")
        self.encoder = Spe_Spa_Attenres_Encoder(in_shape=in_shape, out_embedding=out_embedding)
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=1024)

    def _freeze_encoder(self):
        super()._freeze_encoder()
        #    这里自由解冻其他层
        # for param in self.encoder.res_block6.parameters():
        #     param.requires_grad = True
        # print('最外层已解冻')

    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

MODEL_DICT = {
    'SRACN':Constrastive_learning_Model,
    'Shallow_1DCNN':Shallow_1DCNN,
    'Shallow_3DCNN':Shallow_3DCNN,
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net
}
DATASET_DICT = {
    'SRACN':Dataset_3D,
    'Shallow_1DCNN':Dataset_1D,
    'Shallow_3DCNN':Dataset_3D,
    "Res_3D_18Net": Dataset_3D,
    "Res_3D_34Net": Dataset_3D,
    "Res_3D_50Net": Dataset_3D
}