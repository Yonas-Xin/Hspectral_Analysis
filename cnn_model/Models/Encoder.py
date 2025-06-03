import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class Spe_Spa_Attenres_Encoder(nn.Module):
    '''5个残差块和一个卷积块'''
    def __init__(self,out_embedding=24, in_shape=(138,17,17)):
        super().__init__()
        bands, H, W = in_shape
        self.spectral_attention = ECA_SpectralAttention_3d(138, 2,1)# 光谱注意力
        self.conv_block = Common_3d(1,64,(7,1,1),(3,0,0), 1) # 光谱方向卷积
        self.res_block1 = Residual_block(64,64,(3,3,3),(1,1,1),1)
        self.res_block2 = Residual_block(64,128,(3,3,3),(1,1,1),2)
        self.res_block3 = Residual_block(128,256,(3,3,3),(1,1,1),1)
        self.res_block4 = Residual_block(256,512,(3,3,3),(1,1,1),2)
        self.res_block5 = Residual_block(512,512,(3,3,3),(1,1,1),1)
        self.avg_pool = nn.AvgPool3d(2)
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
        return self.linear(x) # 返回注意力数据
    

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

class SpectralSpatialCNN(nn.Module):
    def __init__(self, spectral_channels, spatial_channels, out_features=128):
        """out_features: 输出特征数"""
        super(SpectralSpatialCNN, self).__init__()
        self.spectral_cnn = SpectralCNN(spectral_channels)
        self.space_cnn = SpaceCNN(spatial_channels)

        self.fc1 = nn.Linear(18048, out_features)
    def forward(self, spectral, spatial):
        spectral = self.spectral_cnn(spectral)
        spatial = self.space_cnn(spatial)
        return self.fc1(torch.cat((spatial, spectral), dim=1))

class SpectralCNN(nn.Module):
    def __init__(self, input_channels):
        super(SpectralCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        """输入：(batch_size, channels, width)"""
        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        x = F.relu(self.conv3(x),inplace=True)
        x = F.relu(self.conv4(x),inplace=True)
        x = self.pool(self.conv5(x))
        x = x.view(x.size(0), -1)  # 展平
        return x

class SpaceCNN(nn.Module):
    def __init__(self, input_channels):
        super(SpaceCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        """输入：(batch_size, channels, height, width)"""
        x = F.relu(self.conv1(x),inplace=True)
        x1 = F.relu(self.conv2(x),inplace=True)
        x1 = F.relu(self.conv3(x1),inplace=True)
        x1 = F.relu(self.conv4(x1),inplace=True)
        x1 = self.pool(self.conv5(x1+x))
        x1 = x1.reshape(x1.size(0), -1)  # 展平
        return x1

if __name__ == '__main__':
    x = torch.randn((1,1,138,17,17))