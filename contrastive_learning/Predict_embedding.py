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
    model_path = r'D:\Programing\pythonProject\Hspectral_Analysis\contrastive_learning\_results\models_pth\SSAR_GF5_202506131106.pth'
    dataset_paths = read_txt_to_list(r'D:\Data\Hgy\龚鑫涛试验数据\program_data\clip_data\.datasets.txt')
    device = torch.device('cuda')

    dataset = SSF_3D(dataset_paths)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=24)
    model = Spe_Spa_Attenres(24, dataset.data_shape)  # 模型实例化
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict['model'])
    frame = Contrasive_learning_predict_frame(device=device)
    out_embeddings = frame.predict(model, dataloader)
    save_matrix_to_csv(out_embeddings, r'D:\Data\Hgy\research_clip_samples\embeddings.csv')