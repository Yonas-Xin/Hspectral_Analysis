import torch.nn as nn
import torch.nn.functional as F
import torch
import math
class ResNet_3D(nn.Module):
    def __init__(self, block, layers, num_classes=1024):
        self.inplanes = 64
        super(ResNet_3D, self).__init__()
        # 网络输入部分
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2,1,1), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        # 中间卷积部分
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2,1,1))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2,1,1))
        self.layer4 = self._make_layer(block, 512, layers[3], stride=(2,1,1))
        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
 
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Spe_Spa_Attenres_Encoder(nn.Module):
    '''6个残差块和一个卷积块'''
    def __init__(self, in_shape, out_embedding=1024):
        super().__init__()
        bands, H, W = in_shape
        self.spectral_attention = ECA_SpectralAttention_3d(bands, 2, 1)# 光谱注意力
        self.conv_block = Common_3d(1, 64, 7, stride=(2,1,1), padding=(3))
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.res_block1 = Residual_block(64, 64, (3,3,3), (1,1,1), 1)
        self.res_block2 = Residual_block(64, 128, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.res_block3 = Residual_block(128, 128, (3,3,3), (1,1,1), 1)
        self.res_block4 = Residual_block(128, 256, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.res_block5 = Residual_block(256, 256, (3,3,3), (1,1,1), 1)
        self.res_block6 = Residual_block(256, 512, (3,3,3), (1,1,1), (2,1,1)) # stride=2
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1)) # 立方体压缩
        self.fc = nn.Linear(512, out_features=out_embedding)
    def forward(self, x):
        x = self.spectral_attention(x)
        x = self.pool(self.conv_block(x))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.avg_pool(self.res_block6(x))
        x = x.view(x.shape[0], -1)
        return self.fc(x)
    
    @property # 返回解冻计划
    def get_unfreeze_plan(self):
        UNFREEZE_PLAN = {80:'res_block4',
                         60:'res_block5',
                         40:'res_block6',
                         20:'linear'} # epoch为20时解冻线性层
        return UNFREEZE_PLAN

class Shallow_3DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=1024):
        super().__init__()
        self.conv1 = Common_3d(1, 64, kernel_size=7, padding=3, stride=(2,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2_1 = Common_3d(64, 64, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv2_2 = Common_3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1))
        self.conv3_1 = Common_3d(128, 128, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv3_2 = Common_3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1))
        self.conv4_1 = Common_3d(256, 256, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv4_2 = Common_3d(256, 512, kernel_size=(3,3,3), padding=(1,1,1), stride=(2,1,1))
        self.pool3 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, out_features=out_embedding)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.conv2_2(self.conv2_1(x))
        x = self.conv3_2(self.conv3_1(x))
        x = self.pool3(self.conv4_2(self.conv4_1(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Shallow_1DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=1024):
        super().__init__()
        self.conv1 = Common_1d(1, 64, kernel_size=7, padding=3, stride=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = Common_1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = Common_1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = Common_1d(256, 512, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(512, out_embedding)

    def forward(self, x):
        # 输入尺寸 [B, 1, L]
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv4(self.conv3(x))
        x = self.pool3(x)         # [B, 128, 1]
        x = x.view(x.size(0), -1) # [B, 128]
        return self.fc(x)

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
 
class Basic_Residual_block(nn.Module):
    """基础残差块"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic_Residual_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample   #对输入特征图大小进行减半处理
        self.stride = stride
 
    def forward(self, x):
        residual = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

class Bottleneck_Residual_block(nn.Module):
    """瓶颈残差块"""
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Residual_block, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
 
        out = self.conv3(out)
        out = self.bn3(out)
 
        if self.downsample is not None:
            residual = self.downsample(x)
 
        out += residual
        out = self.relu(out)
        return out

# ============ECA 光谱注意力组件============
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