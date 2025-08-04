import torch
from cnn_model.Models.Encoder import *
from contrastive_learning.Models.Decoder import *

class Spe_Spa_Attenres(nn.Module):
    def __init__(self, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = Spe_Spa_Attenres_Encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = Spe_Spa_Atten_Decoder(out_embedding, 128, mid_channels=128)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Contra_Res18(nn.Module):
    def __init__(self, out_embedding=1024, in_shape=None):
        super().__init__()
        self.encoder = ResNet_3D(block=Basic_Residual_block, layers=[2,2,2,2], num_classes=out_embedding) # 3d卷积残差编码器
        self.decoder = Spe_Spa_Atten_Decoder(out_embedding, 128, mid_channels=128)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)