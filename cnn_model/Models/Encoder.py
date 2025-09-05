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

class ResNet_2D(nn.Module):
    def __init__(self, block, layers, out_embedding=128, in_shape=None):
        bands, _, _ = in_shape
        self.inplanes = 64
        super(ResNet_2D, self).__init__()
        # 网络输入部分
        self.conv1 = nn.Conv2d(bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 中间卷积部分
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1,1,1))
        self.fc = nn.Linear(512 * block.expansion, out_embedding)
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
 
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
 
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        if x.shape[2] < 16 or x.shape[3] < 16:
            raise ValueError(f"Input H and W must be at least 16, but got H:{x.shape[2]}, W:{x.shape[3]}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if x.shape[2] < 2 or x.shape[3] < 2:
            pass
        else:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SRACN_Encoder(nn.Module): # 以前训练的时候这里的类名为: Spe_Spa_Attenres_Encoder
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

class Common_3DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=1024):
        super().__init__()
        self.conv1 = Common_3d(1, 64, kernel_size=7, padding=3, stride=(2,1,1))
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = Common_3d(64, 128, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv3 = Common_3d(128, 256, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.conv4 = Common_3d(256, 512, kernel_size=(3,3,3), padding=(1,1,1), stride=1)
        self.pool2 = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512, out_features=out_embedding)
    
    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool2(self.conv4(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Common_1DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=1024):
        super().__init__()
        self.conv1 = Common_1d(1, 64, kernel_size=7, padding=3, stride=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2_1 = Common_1d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Common_1d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = Common_1d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Common_1d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = Common_1d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = Common_1d(512, 512, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool1d((1))

        self.fc = nn.Linear(512, out_embedding)

    def forward(self, x):
        # 输入尺寸 [B, 1, L]
        x = self.conv1(x)
        x = self.conv2_2(self.conv2_1(x))
        x = self.conv3_2(self.conv3_1(x))
        x = self.conv4_2(self.conv4_1(x))
        x = self.pool3(x)         # [B, 128, 1]
        x = x.view(x.size(0), -1) # [B, 128]
        return self.fc(x)
    
class Common_2DCNN_Encoder(nn.Module):
    def __init__(self, out_embedding=128, in_shape=None):
        super().__init__()
        bands, _, _ = in_shape
        self.conv1 = Common_2d(bands, 64, kernel_size=7, padding=3, stride=1)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2_1 = Common_2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = Common_2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = Common_2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = Common_2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = Common_2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = Common_2d(512, 512, kernel_size=3, padding=1)
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, out_embedding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_2(self.conv2_1(x))
        x = self.conv3_2(self.conv3_1(x))
        x = self.conv4_2(self.conv4_1(x))
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

class Basic_Residual_block_2d(nn.Module):
    """基础残差块"""
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Basic_Residual_block_2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
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

class Bottleneck_Residual_block_2d(nn.Module):
    """瓶颈残差块"""
    expansion = 4
 
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_Residual_block_2d, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
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

# ============2D CNN组件============
class Common_2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        '''先batch，后激活'''
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
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


# ============其他论文编码器组件============
class HybridSN_encoder(nn.Module):
  """code from: https://github.com/gokriznastic/HybridSN
  自适应输入维度"""
  def __init__(self, out_embedding=256, in_shape=None):
    super(HybridSN_encoder, self).__init__()
    bands, h, w = in_shape
    self.conv1 = nn.Conv3d(1, 8, (7, 3, 3))
    self.conv2 = nn.Conv3d(8, 16, (5, 3, 3))
    self.conv3 = nn.Conv3d(16, 32, (3, 3, 3))
    bands = bands - 12
    self.conv3_2d = nn.Conv2d(bands * 32, 64, (3,3))
    h = h - 8
    # 全连接层（256个节点）
    self.fc =  nn.Linear(h*h*64, out_embedding)
    self.relu = nn.ReLU()

  def forward(self, x):
    if x.dim() == 4:
        x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
    elif x.dim() != 5:
        raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
    out = self.relu(self.conv1(x))
    out = self.relu(self.conv2(out))
    out = self.relu(self.conv3(out))
    # 进行二维卷积，因此把前面的 32*18 reshape 一下，得到 （576, 19, 19）
    out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])
    out = self.relu(self.conv3_2d(out))
    # flatten 操作，变为 18496 维的向量，
    out = out.view(out.size(0), -1)
    out = self.fc(out)
    return out

class Vgg16_encoder(nn.Module):
    """code from: https://github.com/Lornatang/VGG-PyTorch
    为了适应小patch数据, 做了点pool的小修改"""
    def __init__(self, out_embedding=None, in_shape=None):
        super().__init__()
        bands, h, _ = in_shape
        if h < 16:
            raise ValueError("For Vgg16, the input patch size should be at least 16x16.")
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=bands,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,stride=1,padding=1), #(32-3+2)/1+1=32    32*32*64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2,stride=2)   #(32-2)/2+1=16         16*16*64
        )

        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1),  #(16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,stride=1,padding=1), #(16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(16-2)/2+1=8     8*8*128
        )

        self.layer3=nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),


            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=1,padding=1),  #(8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)     #(8-2)/2+1=4      4*4*256
        )

        self.layer4=nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1),  #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2,2)    #(4-2)/2+1=2     2*2*512
        )

        self.layer5=nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),   #(2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,stride=1,padding=1),  #(2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1,1))   #(2-2)/2+1=1      1*1*512
        )

        self.conv=nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )
        self.fc = nn.Linear(512,512)

    def forward(self,x):
        x=self.conv(x)
        x = x.view(-1, 512)
        x=self.fc(x)
        return x


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
class SSRN_encoder(nn.Module):
    """code form: https://github.com/zilongzhong/SSRN"""
    def __init__(self, out_embedding=128, in_shape=None):
        super().__init__()
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

        self.fc = nn.Linear(28, out_embedding)

    def forward(self, x):

        x = F.leaky_relu(self.layer1(x)) #self.bn1(F.leaky_relu(self.layer1(x)))

        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn4(F.leaky_relu(self.layer4(x)))
        x = self.layer5(x)
        x = self.layer6(x)

        x = F.avg_pool2d(x, x.size()[-1])
        x = self.fc(x.squeeze())
        
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()

        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = self.relu(x)
        return x


class MobileNetV1_encoder(nn.Module):
    """code from: https://developer.aliyun.com/article/1309561
       论文地址: https://arxiv.org/abs/1704.04861"""
    def __init__(self, out_embedding=128, in_shape=None):
        super(MobileNetV1_encoder, self).__init__()
        bands, h, w = in_shape

        self.conv1 = nn.Conv2d(bands, 32, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.dw_separable_conv1 = DepthwiseSeparableConv(32, 64)
        self.dw_separable_conv2 = DepthwiseSeparableConv(64, 128)
        self.dw_separable_conv3 = DepthwiseSeparableConv(128, 128)
        self.dw_separable_conv4 = DepthwiseSeparableConv(128, 256)
        self.dw_separable_conv5 = DepthwiseSeparableConv(256, 256)
        self.dw_separable_conv6 = DepthwiseSeparableConv(256, 512)
        self.dw_separable_conv7 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv8 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv9 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv10 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv11 = DepthwiseSeparableConv(512, 512)
        self.dw_separable_conv12 = DepthwiseSeparableConv(512, 1024)
        self.dw_separable_conv13 = DepthwiseSeparableConv(1024, 1024)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, out_embedding)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)

        x = self.dw_separable_conv1(x)
        x = self.dw_separable_conv2(x)
        x = self.dw_separable_conv3(x)
        x = self.dw_separable_conv4(x)
        x = self.dw_separable_conv5(x)
        x = self.dw_separable_conv6(x)
        x = self.dw_separable_conv7(x)
        x = self.dw_separable_conv8(x)
        x = self.dw_separable_conv9(x)
        x = self.dw_separable_conv10(x)
        x = self.dw_separable_conv11(x)
        x = self.dw_separable_conv12(x)
        x = self.dw_separable_conv13(x)

        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
 
        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
 
        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x) if self.stride == 1 else out
        return out
 
 
class MobileNetV2_encoder(nn.Module):
    def __init__(self, out_embedding=128, in_shape=None):
        super(MobileNetV2_encoder, self).__init__()
 
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        bands, _, _ = in_shape
        self.conv1 = nn.Conv2d(bands, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, out_embedding)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
 
    def _make_layers(self, in_planes):
        layers = []
        for t, c, n, s in self.cfgs:
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(Block(in_planes, c, t, stride))
                in_planes = c
        return nn.Sequential(*layers)
 
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

if __name__ == '__main__':
    device = torch.device('cuda')
    model = Unet_3DCNN_Encoder(24, in_shape=(290, 17, 17))
    model.to(device)
    x = torch.randn(1, 1, 290, 17, 17)
    x = x.to(device)
    out = model(x)
    print(out.shape)