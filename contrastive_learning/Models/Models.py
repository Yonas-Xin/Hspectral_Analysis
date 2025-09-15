import torch.nn as nn
import torch.nn.functional as F
from contrastive_learning.Models.Encoder import Contrastive_Model
import torch
    
class Ete_Model(nn.Module):
    def __init__(self,
                 encoder_model_name : str,
                 out_embedding: int = 1024,
                 in_shape : tuple = None, 
                 K: int = 65536, # 为了与Moco初始化对应，这里不管
                 m: float = 0.999,
                 T=0.07):
        super().__init__()  
        self.encoder_q = Contrastive_Model(encoder_model_name=encoder_model_name, out_embedding=out_embedding, in_shape=in_shape)
        self.encoder_k = Contrastive_Model(encoder_model_name=encoder_model_name, out_embedding=out_embedding, in_shape=in_shape)
        self.T = T
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # 确保k与q的初始参数一致，k模型不反向传播参数

    @torch.no_grad()
    def _update_key_encoder(self) -> None:
        """
        Momentum update of the key encoder: k的参数动量更新
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_q.data # 更新k参数

    def forward(self, input_q, input_k):
        q = self.encoder_q(input_q)
        q = nn.functional.normalize(q, dim=1)
        with torch.no_grad():
            self._update_key_encoder()
            k = self.encoder_k(input_k)
            k = nn.functional.normalize(k, dim=1)
        similarity_matrix = torch.matmul(q, k.T)
        mask = torch.eye(input_q.shape[0], dtype=torch.bool).to(input_q.device)
        positives = similarity_matrix[mask].view(input_q.shape[0], -1) # 对角线位置为正样本对
        negatives = similarity_matrix[~mask].view(input_q.shape[0], -1)
        logits = torch.cat([positives, negatives], dim=1)  # 拼接
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input_q.device)
        return logits, labels

class Moco_Model(nn.Module): # 单GPU训练的Moco框架
    def __init__(self, 
                 encoder_model_name: str, 
                 out_embedding: int = 1024, 
                 in_shape: tuple = None,
                 K: int = 65536,
                 m: float = 0.999,
                 T: float = 0.7):
        super().__init__()
        self.K = K
        self.T = T
        self.m = m

        self.encoder_q = Contrastive_Model(encoder_model_name=encoder_model_name, out_embedding=out_embedding, in_shape=in_shape)
        self.encoder_k = Contrastive_Model(encoder_model_name=encoder_model_name, out_embedding=out_embedding, in_shape=in_shape)
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # 确保k与q的初始参数一致，k模型不反向传播参数

        # create the queue
        self.register_buffer("queue", torch.randn(128, K)) # 模型输出维度默认为128. 所以这里写128
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input_q.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels