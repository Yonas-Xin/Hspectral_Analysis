import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import numpy as np
from contrastive_learning.Models.Data import SSF_3D
from torch.utils.data import DataLoader
from contrastive_learning.Models.Models import Spe_Spa_Attenres
from utils import read_txt_to_list, save_matrix_to_csv
from tqdm import tqdm
from contrastive_learning.Models.Frame import Contrasive_learning_predict_frame
if __name__ == '__main__':
    model_path = r'C:\Users\85002\Desktop\模型\Spe_Spa_Attenres110_retrain_202504281258.pth'
    dataset_paths = read_txt_to_list(r'D:\Data\Hgy\research_clip_samples\.datasets.txt')
    device = torch.device('cuda')

    model = Spe_Spa_Attenres(24, in_shape=(138,17,17))
    state_dict = torch.load(model_path, weights_only=True, map_location='cuda:0')
    model.load_state_dict(state_dict['model'])

    dataset = SSF_3D(dataset_paths)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=24)

    frame = Contrasive_learning_predict_frame(device=device)
    out_embeddings = frame.predict(model, dataloader)
    save_matrix_to_csv(out_embeddings, r'D:\Data\Hgy\research_clip_samples\embeddings.csv')