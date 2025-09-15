import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import numpy as np
from contrastive_learning.Models.Data import Contrastive_Dataset, DynamicCropDataset
from torch.utils.data import DataLoader
from contrastive_learning.Models.Models import Ete_Model
from utils import read_txt_to_list, save_matrix_to_csv
from tqdm import tqdm
from contrastive_learning.Models.Frame import Contrasive_learning_predict_frame

if __name__ == '__main__':
    model_path = r'C:\Users\85002\Desktop\模型\模型pth与log\Spe_Spa_Attenres110_retrain_202504281258.pth'
    input_img = r'C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat'
    input_shp = r'C:\Users\85002\Desktop\TempDIR\out.shp'
    device = torch.device('cuda')

    dataset = DynamicCropDataset(input_img, input_shp, block_size=17)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=24)
    model = Ete_Model(24, dataset.data_shape)  # 模型实例化
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    model.load_state_dict(state_dict['model'])
    frame = Contrasive_learning_predict_frame(device=device)
    out_embeddings = frame.predict(model, dataloader)
    save_matrix_to_csv(out_embeddings, r'D:\Data\Hgy\research_clip_samples\embeddings.csv')