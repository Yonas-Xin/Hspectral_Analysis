import torch.nn as nn
import torch.nn.functional as F
from contrastive_learning.Models.Encoder import Spe_Spa_Attenres_Encoder
from contrastive_learning.Models.Decoder import Spe_Spa_Atten_Decoder

class Spe_Spa_Attenres(nn.Module):
    def __init__(self, out_embedding=24, in_shape=(138,17,17)):
        super().__init__()  
        self.encoder = Spe_Spa_Attenres_Encoder(out_embedding=out_embedding, in_shape=in_shape)
        self.decoder = Spe_Spa_Atten_Decoder(out_embedding, 128, mid_channels=128)
    def forward(self, x):
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        return embedding, x
    def predict(self, x):
        return self.encoder(x)
