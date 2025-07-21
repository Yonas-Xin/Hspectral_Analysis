import torch
from cnn_model.Models.Encoder import *


if __name__=='__main__':
    x = torch.randn(2,1,25,25,138)