import torch
from cnn_model.models.Encoder import Spe_Spa_Attenres_Encoder


if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)