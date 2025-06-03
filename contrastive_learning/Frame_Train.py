import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from models.Frame import Cl_Frame
from contrastive_learning.models.Data import SSF,SSF_3D,SSF_3D_H5
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from models.Models import Spe_Spa_Attenres
from Feature_transform import HighDimBatchAugment
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR,ExponentialLR,StepLR
from utils import search_files_in_directory, read_txt_to_list
import matplotlib.pyplot as plt
if __name__ == '__main__':
    clean_noise_samples = False # 是否掩膜噪声负样本
    if_full_cpu = True # 是否全负荷cpu
    load_from_ck = False # 从断点处开始训练
    epochs = 30  # epoch
    batch = 4  # batch
    init_lr = 1e-4  # lr
    min_lr = 1e-7 # 最低学习率
    warmup_epochs = 0
    model = Spe_Spa_Attenres(out_embedding=24, in_shape=(138, 17, 17))
    config_model_name = "Spe_Spa_Atten_pretrain"  # 模型名称
    images_dir = r'D:\Data\Hgy\预处理\contrastive_learning_17x17\.datasets.txt' # 数据集
    ck_pth = r'D:\Programing\pythonProject\Hyperspectral_Analysis\contrastive_learning\models\Spe_Spa_Atten_pretrain_202504252344.pth' # 保存的权重文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置

    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1) # 学习率调度器
    augment = HighDimBatchAugment(crop_size=(17, 17))  # 数据特征转换

    # 配置dataloader
    image_lists = read_txt_to_list(images_dir)
    dataset = SSF_3D(image_lists)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2,
                            persistent_workers=True)  # 数据迭代器

    frame = Cl_Frame(augment=augment, model_name=config_model_name, epochs=epochs, min_lr=min_lr, warmup_epochs=warmup_epochs,
                     device=device, if_full_cpu=if_full_cpu)
    frame.train(model=model, optimizer=optimizer,scheduler=scheduler,dataloader=dataloader,
                ck_pth=ck_pth, clean_noise_samples=clean_noise_samples, clean_th=0.99, load_from_ck=load_from_ck)