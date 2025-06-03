from models.Encoder import *
from models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch
class Constrastive_learning_Model(nn.Module):
    def __init__(self, out_embedding=24, out_classes=8, in_shape=(138,17,17)):
        super().__init__()
        self.encoder = Spe_Spa_Attenres_Encoder(out_embedding=out_embedding, in_shape=in_shape) # 3d卷积残差编码器
        self.decoder = deep_classfier(out_embedding, out_classes, mid_channels=4096)
    def forward(self, x):
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
