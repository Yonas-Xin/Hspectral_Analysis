import torch.nn as nn
import torch.nn.functional as F
from contrastive_learning.Models.Encoder import *
from contrastive_learning.Models.Decoder import *
from cnn_model.Models.Models import Res_3D_18Net
import torch

class Spe_Spa_Attenres(nn.Module):
    def __init__(self, out_embedding=24, in_shape=(138,17,17)):
        super().__init__()  
        self.encoder = Spe_Spa_Attenres_Encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = Spe_Spa_Atten_Decoder(out_embedding, 128, mid_channels=128)
    def forward(self, x):
        if x.dim() == 4:
            x = x.unsqueeze(1)  # 增加一个维度到 [B, 1, C, H, W]
        elif x.dim() != 5:
            raise ValueError(f"Expected input dimension 4 or 5, but got {x.dim()}")
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        return embedding, x
    def predict(self, x):
        return self.encoder(x)

class Contra_Res18(nn.Module):
    def __init__(self, out_embedding=1024, in_shape=(138,17,17)):
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

class Moco3D(nn.Module): # 单GPU训练的Moco框架
    def __init__(self, 
                 base_model: nn.Module = Contra_Res18, 
                 out_embedding: int = 1024, 
                 in_shape: tuple = (138, 17, 17),
                 dim: int = 128,
                 K: int = 65536,
                 m: float = 0.999,
                 T: float = 0.7):
        super().__init__()
        self.K = K
        self.T = T
        self.m = m

        self.encoder_q = base_model(out_embedding=out_embedding, in_shape=in_shape)
        self.encoder_k = base_model(out_embedding=out_embedding, in_shape=in_shape)
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # 确保k与q的初始参数一致，k模型不反向传播参数

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
    @torch.no_grad()
    def _momentum_update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder: k的参数动量更新
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys) -> None:
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input_q, input_k):
        """
        Input:
            input_q: a batch of query images
            input_k: a batch of key images
        Output:
            logits, targets
        """
        # compute query features
        q = self.encoder_q(input_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1) 
        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k = self.encoder_k(input_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1) # 计算批次中每每两个正样本对的相似度
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()]) # 计算批次中样本与队列中所有负样本的相似度

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels