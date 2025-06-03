import torch.nn as nn
import torch

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.mask = None

    def cosine_similarity_matrix(self,x, th=0.9):
        """
        根据余弦相似度计算掩膜的mask
        Args:
            x: Tensor of shape (nums, dims)
        Returns:
            sim_matrix: Tensor of shape (nums, nums)
        """
        # 归一化（L2范数）
        x_norm = x / torch.norm(x, p=2, dim=1, keepdim=True)
        # 计算相似性矩阵
        sim_matrix = torch.mm(x_norm, x_norm.T)  # (nums, nums)
        mask = (~(sim_matrix > th)).float().to(x.device)
        self.mask = mask # 0 值代表被掩膜的值，1值代表需要计算的值

    def forward(self, input):
        '''
        input(2*batch, dims),前batch和后batch分属于两部分增强的数据
        对一个batch为3的训练样本来说，样本增强得到A,B,C,a,b,c。Aa，Bb，Cc分别是正样本对
        feature(2*batch, dims)，前batch与后batch分别为两个增强样本，即ABC与abc。
        在计算损失时，与传统infoNCE不同的是，例如，对A而言，传统infoNCE将Ab,Ac视作负样本对集，Aa视作正样本对。
        而在该函数中，AB，AC都视作为负样本对集。因此，构建的logit应该为（Aa，AB，AC，Ab，Ac），其中AA予以对角线消除。
        对于一个batch而言，总logit的形状应该为（2*batch, 2*batch-1）。 ps：标准计算公式里，logit的形状为（batch，batch）
        '''
        batch_size = input.shape[0]//2
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # label转独热编码
        labels = labels.to(input.device)
        if self.mask is not None:
            self.mask[labels==1] = 1 # 正样本位置变为1

        input = nn.functional.normalize(input, dim=1)
        similarity_matrix = torch.matmul(input, input.T) / self.temperature#L2+点集等于计算余弦度，构建相似矩阵

        # 消除对角线自匹配，即删除AA，BB，CC的相似度
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(input.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        if self.mask is not None:
            self.mask = self.mask[~mask].view(similarity_matrix.shape[0], -1) # 消除对角线

        # 构建pos/neg样本,即pos[Aa,Bb，Cc], neg[[AB,AC,Ab,Ac],[BA,BC,......]......]
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # 正样本对
        negatives = similarity_matrix[~labels.bool()].view(labels.shape[0], -1) # 负样本对集
        if self.mask is not None:
            self.mask = torch.cat([self.mask[labels.bool()].view(labels.shape[0], -1),self.mask[~labels.bool()].view(labels.shape[0], -1)],
                                  dim=1) # mask位置也做拼接

        logits = torch.cat([positives, negatives], dim=1) # 拼接正负样本对，正样本位于位置0，即[[Aa,AB,AC,Ab,Ac],[Bb,BA,......]......]
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(input.device)# 初始化一个全为0的label，进行交叉熵计算式会把第0个值当作真实值
        return cross_entropy_loss_with_mask(logits, labels, self.mask)

def cross_entropy_loss_with_mask(logits, targets, mask=None):
    '''使用mask掩膜的交叉熵函数'''
    # Step 1: Softmax
    max_logits = logits.max(dim=1, keepdim=True).values  # 避免数值溢出
    exp_logits = torch.exp(logits - max_logits)  # 数值稳定版 Softmax, 掩膜
    if mask is not None:
        exp_logits = exp_logits*mask # 乘上一个mask，组织被掩膜的数据梯度传播（被掩膜数值梯度为0）
    probs = exp_logits / exp_logits.sum(dim=1, keepdim=True)  # (batch_size, num_classes)
    # Step 2: 负对数似然（NLL）
    batch_size = logits.shape[0]
    true_class_probs = probs[torch.arange(batch_size), targets]  # 选取真实类别的概率
    loss = -torch.log(true_class_probs)  # (batch_size,)
    return loss.mean()

if __name__ == "__main__":
    pass