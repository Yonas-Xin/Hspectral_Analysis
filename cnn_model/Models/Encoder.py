import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class Spe_Spa_Attenres_Encoder(nn.Module):
    '''5个残差块和一个卷积块'''
    def __init__(self, out_embedding=24, in_shape=(138,17,17)):
        super().__init__()
        bands, H, W = in_shape
        self.spectral_attention = ECA_SpectralAttention_3d(bands, 2,1)# 光谱注意力
        self.conv_block = Common_3d(1,64,(7,1,1),(3,0,0), 1) # 光谱方向卷积
        self.res_block1 = Residual_block(64,64,(3,3,3),(1,1,1),1)
        self.res_block2 = Residual_block(64,128,(3,3,3),(1,1,1),2) # stride=2
        self.res_block3 = Residual_block(128,256,(3,3,3),(1,1,1),1)
        self.res_block4 = Residual_block(256,512,(3,3,3),(1,1,1),2) # stride=2
        self.res_block5 = Residual_block(512,512,(3,3,3),(1,1,1),1)
        self.avg_pool = nn.AvgPool3d(2) # 立方体压缩
        in_feature = int(bands/8)*int(H/8)*int(W/8)*512
        self.linear = nn.Linear(in_feature, out_features=out_embedding)
    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.conv_block(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.avg_pool(self.res_block5(x))
        x = x.view(x.shape[0], -1)
        return self.linear(x)
    
    @property # 返回解冻计划
    def get_unfreeze_plan(self):
        UNFREEZE_PLAN = {80:'res_block3',
                         60:'res_block4',
                         40:'res_block5',
                         20:'linear'} # epoch为20时解冻线性层
        return UNFREEZE_PLAN

class Shallow_3DCNN_Encoder(nn.Module):
    def __init__(self, in_shape=(138,17,17)):
        super().__init__()
        bands, H, W = in_shape
        self.conv1 = Common_3d(1, 32, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2_1 = Common_3d(32, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2_2 = Common_3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv3 = Common_3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv4 = Common_3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1))
        self.pool3 = nn.AvgPool3d(2)
        in_feature = int(bands/8)*int(H/8)*int(W/8)*128
        self.linear = nn.Linear(in_feature, out_features=128)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv4(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.linear(x)

class Shallow_1DCNN_Encoder(nn.Module):
    def __init__(self, in_shape=(138,)):
        super().__init__()
        channels = 1
        length = in_shape[0]

        self.conv1 = Common_1d(channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2_1 = Common_1d(32, 64, kernel_size=3, padding=1)
        self.conv2_2 = Common_1d(64, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = Common_1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = Common_1d(128, 128, kernel_size=3, padding=1)
        self.pool3 = nn.AvgPool1d(2)

        in_feature = int(length/8) * 128
        self.linear = nn.Linear(in_feature, 128)

    def forward(self, x):
        # 输入尺寸 [B, 1, L]
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.conv4(self.conv3(x))
        x = self.pool3(x)         # [B, 128, 1]
        x = x.view(x.size(0), -1) # [B, 128]
        return self.linear(x)
    
# class Unet_3DCNN_Encoder(nn.Module):
#     def __init__(self, out_embedding=24, in_shape=(138,17,17)):
#         super().__init__()
#         bands, H, W = in_shape
#         self.spectral_attention = ECA_SpectralAttention_3d(bands, 2,1)# 光谱注意力
#         self.conv1_1 = Common_3d(1, 64, (3,3,3), (1,1,1), 1)
#         self.conv1_2 = Common_3d(64, 64, (3,3,3), (1,1,1), 1)
#         self.pool1 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)) # 只针对光谱方向压缩
#         self.conv1_1x1 = Common_3d(64, 128, (1,1,1), (0,0,0), 1)
#         bands_1  = int(bands/2)

#         self.conv2_1 = Common_3d(64, 128, (3,3,3), (1,1,1), 1)
#         self.conv2_2 = Common_3d(128, 128, (3,3,3), (1,1,1), 1)
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
#         self.conv2_1x1 = Common_3d(128, 256, (1,1,1), (0,0,0), 1)
#         bands_2 = int(bands_1/2)

#         self.conv3_1 = Common_3d(128, 256, (3,3,3), (1,1,1), 1)
#         self.conv3_2 = Common_3d(256, 256, (3,3,3), (1,1,1), 1)
#         self.pool3 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
#         self.conv3_1x1 = Common_3d(256, 512, (1,1,1), (0,0,0), 1)
#         bands_3 = int(bands_2/2)

#         self.conv4_1 = Common_3d(256, 512, (3,3,3), (1,1,1), 1)
#         self.conv4_2 = Common_3d(512, 512, (3,3,3), (1,1,1), 1)
#         self.pool4 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
#         bands_4 = int(bands_3/2)

#         self.conv5_1 = Common_3d(512, 256, (3,3,3), (1,1,1), 1)
#         self.conv5_2 = Common_3d(256, 256, (3,3,3), (1,1,1), 1)
#         self.pool5 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
#         bands_5 = int((bands_4+bands_3)/2)

#         self.conv6_1 = Common_3d(256, 128, (3,3,3), (1,1,1), 1)
#         self.conv6_2 = Common_3d(128, 128, (3,3,3), (1,1,1), 1)
#         self.pool6 = nn.MaxPool3d(kernel_size=(2,1,1), stride=(2, 1, 1))
#         bands_6 = int((bands_5+bands_2)/2)

#         self.conv7_1 = Common_3d(128, 64, (3,3,3), (1,1,1), 1)
#         self.conv7_2 = Common_3d(64, 64, (3,3,3), (1,1,1), 1)
#         self.avg_pool = nn.AvgPool3d(2) # 立方体压缩
#         infeature = int(H/2)* int(W/2)* int((bands_6+bands_1)/2)
#         self.linear = nn.Linear(infeature, out_features=out_embedding)
#     def forward(self, x):
#         x = self.spectral_attention(x)
#         x1 = self.conv1_2(self.conv1_1(x))
#         x1_out = self.conv1_1x1(x1) 
#         x = self.pool1(x1)
#         x1 = self.conv2_2(self.conv2_1(x))
#         x2_out = self.conv2_1x1(x1)
#         x = self.pool2(x1)
#         x = self.conv3_2(self.conv3_1(x))
#         x3_out = self.conv3_1x1(x)
#         x = self.pool3(x)
#         x = self.pool4(self.conv4_2(self.conv4_1(x)))
#         x = self.pool5(self.conv5_2(self.conv5_1(torch.cat((x, x3_out), dim=2))))
#         x = self.pool6(self.conv6_2(self.conv6_1(torch.cat((x, x2_out), dim=2))))
#         x = self.avg_pool(self.conv7_2(self.conv7_1(torch.cat((x, x1_out), dim=2))))
#         x = x.view(x.shape[0], -1)
#         return self.linear(x)

# =================================================================================================
# 编码器组件
# =================================================================================================
class Common_3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,3), padding=(1,1,1), stride=1):
        super(Common_3d,self).__init__()
        '''先batch，后激活'''
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))

class Residual_block(nn.Module):
    '''标准残差块结构'''
    def __init__(self, in_channel=1, out_channel=64,kernel_size=(3,3,3), padding=(1,1,1), stride=1):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channel, out_channel, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm3d(out_channel),
            )
        if in_channel!=out_channel:
            self.use_downsample = True
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channel, out_channel, kernel_size=(1, 1, 1), stride=stride, bias=False),
                nn.BatchNorm3d(out_channel)
            )
        else:self.use_downsample = False
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.bottleneck(x)
        if self.use_downsample:
            x = self.downsample(x)
        return self.relu(out+x)

class SE_SpectralAttention_3d(nn.Module):
    def __init__(self, bands):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, bands))  # 压缩空间维度 (rows, cols) → (1, 1)
        self.fc = nn.Sequential(
            nn.Linear(bands, bands // 8),  # 降维减少计算量
            nn.ReLU(),
            nn.Linear(bands // 8, bands),  # 恢复原始维度
            nn.Sigmoid()  # 输出 [0,1] 的权重
        )
    def forward(self, x):
        # x.shape: (batch, 1, rows, cols, bands)
        batch, _, _, _, bands = x.shape
        gap = self.gap(x)
        gap = gap.view(batch, bands)
        weights = self.fc(gap)
        weights = weights.view(batch, 1, 1, 1, bands)
        return x * weights

class ECA_SpectralAttention_3d(nn.Module):
    def __init__(self, bands,gamma=2,b=1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d((bands,1, 1))  # 压缩空间维度 (rows,cols) → (1,1)
        kernel_size = int(abs((math.log(bands, 2) + b) / gamma))
        if kernel_size%2==0:
            kernel_size+=1
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x.shape: (batch, 1, rows, cols, bands)
        batch, _, bands, _, _ = x.shape
        gap = self.gap(x)  # [batch, 1, 1, 1, bands]
        gap = gap.view(batch, 1, bands)  # [batch, 1, bands]
        attn_weights = self.conv(gap)  # 滑动窗口计算局部光谱关系
        # Sigmoid 归一化到 [0,1]
        attn_weights = self.sigmoid(attn_weights)  # [batch, 1, bands]
        # 恢复形状为 (batch,1,1,1,bands)
        attn_weights = attn_weights.view(batch, 1, bands, 1, 1)
        return x * attn_weights
    
# ============2D resnet组件============
class Bottleneck(nn.Module):
    expansion = 4  # 通道扩张倍率

    def __init__(self, in_channels, mid_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, mid_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(mid_channels * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # identity shortcut是否需要变换
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

# ============1D CNN组件============
class Common_1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        '''先batch，后激活'''
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.batch_norm(self.conv(x)))
    
# ============3D CNN改进组件============
class Spectral_Conv3d(nn.Module):
    def __init__(self, in_channels, out_channels, input_shape, kernel_size=3, padding=1, stride=1):
        super().__init__()
        bands, h, w = input_shape
        self.conv = nn.Conv1d(h*w*in_channels, h*w*out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm1d(h*w*out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels
    def forward(self, x):
        batch, channels, bands, h, w = x.shape
        x = x.permute(0,1,3,4,2)  # 调整维度顺序到 [B, C, H, W, bands]
        x = x.reshape(batch, channels * h * w, bands)  # 展平
        x = self.relu(self.batch_norm(self.conv(x)))
        x = x.reshape(batch, self.out_channels, h, w, -1)
        return x.permute(0,1,4,2,3)

class Spectral_Pool3d(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
    def forward(self, x):
        batch, channels, bands, h, w = x.shape
        x = x.permute(0,1,3,4,2)  # 调整维度顺序到 [B, C, H, W, bands]
        x = x.reshape(batch, channels * h * w, bands)  # 展平
        x = self.pool(x)
        x = x.reshape(batch, channels, h, w, -1)
        return x.permute(0,1,4,2,3)
class Unet_3DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=128, in_shape=(138,17,17)):
        super().__init__()
        bands, H, W = in_shape
        # self.spectral_attention = ECA_SpectralAttention_3d(bands, 2,1)# 光谱注意力
        self.start_conv = Common_3d(1, 1, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv1_1 = Spectral_Conv3d(1, 2, in_shape, 3, 1, 1)
        self.conv1_2 = Spectral_Conv3d(2, 2, in_shape, 3, 1, 1)
        self.pool1 = Spectral_Pool3d(kernel_size=2, stride=2) # 只针对光谱方向压缩
        in_shape = (int(bands/2), H, W)

        self.conv2_1 = Spectral_Conv3d(2, 4, in_shape, 3, 1, 1)
        self.conv2_2 = Spectral_Conv3d(4, 4, in_shape, 3, 1, 1)
        self.pool2 = Spectral_Pool3d(kernel_size=2, stride=2)
        in_shape = (int(bands/4), H, W)

        self.conv3_1 = Spectral_Conv3d(4, 8, in_shape, 3, 1, 1)
        self.conv3_2 = Spectral_Conv3d(8, 8, in_shape, 3, 1, 1)
        self.pool3 = Spectral_Pool3d(kernel_size=2, stride=2)
        in_shape = (int(bands/8), H, W)

        self.conv4_1 = Spectral_Conv3d(8, 16, in_shape, 3, 1, 1)
        self.conv4_2 = Spectral_Conv3d(16, 16, in_shape, 3, 1, 1)
        self.out_conv = Common_3d(16, 32, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.pool4 = nn.AvgPool3d(kernel_size=2, stride=2)

        in_feature = int(bands/16) * H//2 * W//2 * 32
        self.linear = nn.Linear(in_feature, out_features=out_embedding)
    def forward(self, x):
        # x = self.spectral_attention(x)
        x = self.start_conv(x)
        x = self.pool1(self.conv1_2(self.conv1_1(x)))
        x = self.pool2(self.conv2_2(self.conv2_1(x)))
        x = self.pool3(self.conv3_2(self.conv3_1(x)))
        x = self.pool4(self.out_conv(self.conv4_2(self.conv4_1(x))))
        x = x.view(x.shape[0], -1)
        return self.linear(x)
    
if __name__ == '__main__':
    device = torch.device('cuda')
    model = Unet_3DCNN_Encoder(24, in_shape=(290, 17, 17))
    model.to(device)
    x = torch.randn(1, 1, 290, 17, 17)
    x = x.to(device)
    out = model(x)
    print(out.shape)