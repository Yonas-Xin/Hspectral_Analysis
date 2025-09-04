"""针对Models模块的升级"""
from cnn_model.Models.Encoder import *
from cnn_model.Models.Decoder import *
import torch.nn as nn
import torch.nn.functional as F
import torch

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
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
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
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
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
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
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
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
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
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
        super().__init__()
        self.encoder = Shallow_1DCNN_Encoder(out_embedding=out_embedding)  # 1D CNN 编码器
        self.decoder = deep_classfier(128, out_classes, mid_channels=1024)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, bands]
        elif x.dim() == 4: # [B, c, h, w]
            _, _, h, w = x.shape
            left_top = h // 2 - 1 if h % 2 == 0 else h // 2
            x = x[:, :, left_top, left_top]
            x = x.unsqueeze(1)
        elif x.dim() == 5: # [B, 1, c, h, w]
            x = x.squeeze(1)
            _, _, h, w = x.shape
            left_top = h // 2 - 1 if h % 2 == 0 else h // 2
            x = x[:, :, left_top, left_top]
            x = x.unsqueeze(1)
        elif x.dim() != 3:
            raise ValueError(f"Expected input dimension 2 or 3, but got {x.dim()}")
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class SRACN(My_Model):
    def __init__(self, out_classes, out_embedding=128, in_shape=None):
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


# ==============================其他论文中的模型==============================
class SPCModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(SPCModuleIN, self).__init__()
                
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(7,1,1), stride=(2,1,1), bias=False)
        #self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, input):
        
        input = input.unsqueeze(1)
        
        out = self.s1(input)
        
        return out.squeeze(1) 
class SPAModuleIN(nn.Module):
    def __init__(self, in_channels, out_channels, k=49, bias=True):
        super(SPAModuleIN, self).__init__()
                
        # print('k=',k)
        self.s1 = nn.Conv3d(in_channels, out_channels, kernel_size=(k,3,3), bias=False)
        #self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        # print(input.size())
        out = self.s1(input)
        out = out.squeeze(2)
        # print(out.size)
        
        return out
class ResSPC(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPC, self).__init__()
                
        self.spc1 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm3d(in_channels),)
        
        self.spc2 = nn.Sequential(nn.Conv3d(in_channels, in_channels, kernel_size=(7,1,1), padding=(3,0,0), bias=False),
                                    nn.LeakyReLU(inplace=True),)
        
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, input):
                
        out = self.spc1(input)
        out = self.bn2(self.spc2(out))
        
        return F.leaky_relu(out + input)    
class ResSPA(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(ResSPA, self).__init__()
                
        self.spa1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),
                                    nn.BatchNorm2d(in_channels),)
        
        self.spa2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                                    nn.LeakyReLU(inplace=True),)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
                
        out = self.spa1(input)
        out = self.bn2(self.spa2(out))
        
        return F.leaky_relu(out + input)
class SSRN(My_Model):
    """code form: https://github.com/zilongzhong/SSRN"""
    def __init__(self, out_classes, out_embedding=None, in_shape=None):
        super(SSRN, self).__init__()
        bands, h, w = in_shape
        k = (bands - 6) // 2 # 自动计算k值

        self.layer1 = SPCModuleIN(1, 28)
        #self.bn1 = nn.BatchNorm3d(28)
        
        self.layer2 = ResSPC(28,28)
        
        self.layer3 = ResSPC(28,28)
        
        #self.layer31 = AKM(28, 28, [97,1,1])   
        self.layer4 = SPAModuleIN(28, 28, k=k)
        self.bn4 = nn.BatchNorm2d(28)
        
        self.layer5 = ResSPA(28, 28)
        self.layer6 = ResSPA(28, 28)

        self.fc = nn.Linear(28, out_classes)

    def forward(self, x):

        x = F.leaky_relu(self.layer1(x)) #self.bn1(F.leaky_relu(self.layer1(x)))
        #print(x.size())
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer31(x)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x
MODEL_DICT = {
    'SRACN':SRACN,
    'Shallow_1DCNN':Shallow_1DCNN,
    'Shallow_3DCNN':Shallow_3DCNN,
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net
}