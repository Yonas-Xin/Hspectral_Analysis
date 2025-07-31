import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from Models.Moco_Frame import Moco_Frame,train
from contrastive_learning.Models.Data import Dataset_3D
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from Models.Models import Moco3D
from contrastive_learning.Models.Feature_transform import HighDimBatchAugment
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR,ExponentialLR,StepLR
from utils import search_files_in_directory, read_txt_to_list
import math
from multiprocessing import cpu_count

if __name__ == '__main__':
    if_full_cpu = True # 是否全负荷cpu
    load_from_ck = False # 从断点处开始训练
    epochs = 30  # epoch
    batch = 8  # batch
    init_lr = 1e-4  # lr
    min_lr = 1e-7 # 最低学习率
    config_model_name = "Moco_Res18"  # 模型名称
    images_dir = r'c:\Users\85002\Desktop\TempDIR\ZY-01-Test\clip_by_shpfile' # 数据集
    ck_pth = r'D:\Programing\pythonProject\Hspectral_Analysis\contrastive_learning\_results\models_pth\Moco_Res18_202507221146.pth' # 保存的权重文件
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置

    step_size = epochs // (math.log10(init_lr // min_lr) + 1) # 自动计算学习率调度器的步长
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    # 配置dataloader
    image_lists = search_files_in_directory(images_dir, '.tif')
    dataset = Dataset_3D(image_lists)
    model = Moco3D(out_embedding=24, in_shape=dataset.data_shape, K=1024)  # 模型实例化
    print(f"Image shape: {dataset.data_shape}")
    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1) # 学习率调度器
    if step_size <= 0: # step太小,那么不设置调度器
        scheduler = None
    
    augment = HighDimBatchAugment(crop_size=(dataset.data_shape[1], dataset.data_shape[2]), spectral_mask_p=0.5, band_dropout_prob=0)  # 数据特征转换
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=dataloader_num_workers, 
                            drop_last=True, persistent_workers=True)  # 数据迭代器

    frame = Moco_Frame(augment=augment, 
                       model_name=config_model_name, 
                       epochs=epochs, 
                       min_lr=min_lr, 
                       device=device, 
                       if_full_cpu=if_full_cpu)
    
    train(frame=frame,
          model=model, 
          optimizer=optimizer,
          scheduler=scheduler,
          dataloader=dataloader,
          ck_pth=ck_pth, 
          load_from_ck=load_from_ck)