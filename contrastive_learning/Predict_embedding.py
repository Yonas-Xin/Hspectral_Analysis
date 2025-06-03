import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import numpy as np
from contrastive_learning.models.Data import SSF_3D
from torch.utils.data import DataLoader
from models.Models import Spe_Spa_Attenres
from utils import read_txt_to_list
from tqdm import tqdm
if __name__ == '__main__':
    model_path = r'C:\Users\85002\Desktop\模型\Spe_Spa_Attenres110_retrain_202504281258.pth'
    # model_path = r'D:\Programing\pythonProject\Hyperspectral_Analysis\contrastive_learning\F3FN_3d_pretrain_202504041110.pth'
    dataset_paths = read_txt_to_list(r'D:\Data\Hgy\research_clip_samples\.datasets.txt')
    dataset = SSF_3D(dataset_paths)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=24)

    model = Spe_Spa_Attenres(24, in_shape=(138,17,17))
    state_dict = torch.load(model_path, weights_only=True, map_location='cuda:0')
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda')

    model.to(device)
    data = torch.empty((len(dataset), 24), dtype=torch.float32)  # 预分配内存，加快速度
    with torch.no_grad():
        model.eval()
        idx = 0
        for image in tqdm(dataloader, total=len(dataloader)):
            image = image.unsqueeze(1).to(device)
            predict = model.predict(image)
            data[idx:idx+len(predict)] = predict
            idx += len(predict)
    data = data.cpu().numpy()
    np.savez_compressed(r'SSAR_re50_Embedding24', data = data)
    print("数据已保存")