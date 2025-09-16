import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from contrastive_learning.Models.Frame import Contrastive_Frame, train
from contrastive_learning.Models.Data import Contrastive_Dataset, MIRBS_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from Models.Models import Ete_Model, Moco_Model
from contrastive_learning.Models.Feature_transform import HighDimBatchAugment
from torch.optim.lr_scheduler import StepLR
from utils import search_files_in_directory
import math
from multiprocessing import cpu_count
TRAIN_DICT = {
    "ETE": Ete_Model,
    "MOCO": Moco_Model
}
DATASET_DICT = {
    1: Contrastive_Dataset,
    2: MIRBS_Dataset
}
# 可选用的模型如下：
# 'SRACN' 'Common_1DCNN' 'Common_2DCNN' 'Common_3DCNN' "Res_3D_18Net" "Res_3D_34Net" "Res_3D_50Net" 'SSRN' 
# 'HybridSN' 'Vgg16' 'MobileNetV1' 'MobileNetV2' 'ResNet18' 'ResNet34' 'ResNet50'

encdeor_model_name = 'Common_1DCNN'
config_model_name = "Test"  # 模型名称
TRAIN_MODE = "ETE" # ETE or MOCO 训练方式选择
DATA_MANAGE_MODE = 1 # 数据管理方式，1为根据裁剪的样本进行训练，2为根据选择的影像自动训练
if_full_cpu = True # 是否全负荷cpu
epochs = 100  # epoch
batch = 16  # batch
init_lr = 1e-4  # lr
min_lr = 1e-7 # 最低学习率
out_embedding = 128 # 编码器输出维度
images_dir = r'c:\Users\85002\Desktop\TempDIR\ZY-01-Test\clip_by_shpfile' # 数据集
ck_pth = None

K = 65536
m = 0.999
T = 0.07
patch_size = 17 # DATA_MANAGE_MODE = 2时需指定该
MUITITHREADING_MODE = True
if __name__ == '__main__':  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置
    step_size = epochs // (math.log10(init_lr // min_lr) + 1) # 自动计算学习率调度器的步长
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    dataloader_num_workers = dataloader_num_workers if MUITITHREADING_MODE else 0
    # 配置dataloader
    image_lists = search_files_in_directory(images_dir, ('tif', 'dat'))
    dataset = DATASET_DICT[DATA_MANAGE_MODE](image_lists, patch_size=patch_size, multith_mode=MUITITHREADING_MODE)
    model = TRAIN_DICT[TRAIN_MODE](encoder_model_name=encdeor_model_name, out_embedding=out_embedding, in_shape=dataset.data_shape, K=K, m=m, T=T)  # 模型实例化
    print(f"Image shape: {dataset.data_shape}")
    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1) # 学习率调度器
    if step_size <= 0: # step太小,那么不设置调度器
        scheduler = None

    augment = HighDimBatchAugment(spectral_mask_prob=0.5, band_dropout_prob=0)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=3,
                            persistent_workers=MUITITHREADING_MODE, drop_last=True)  # 数据迭代器

    frame = Contrastive_Frame(augment=augment, 
                           model_name=f'{encdeor_model_name}_{config_model_name}_Emb{out_embedding}', 
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
        )