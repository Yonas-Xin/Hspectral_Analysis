"""针对Train模块的升级版"""
import sys
sys.path.append('.')
import torch
import torch.optim as optim
from cnn_model.Models.Models import SRACN, Common_1DCNN, Common_3DCNN, SSRN, HybridSN, Vgg16_net, MobileNetV1, \
                                    MobileNetV2, Common_2DCNN, ResNet18, Res_3D_18Net, Res_3D_34Net, Res_3D_50Net, \
                                    ResNet34, ResNet50
from cnn_model.Models.Data import CNN_Dataset
from torch.optim.lr_scheduler import StepLR
from cnn_model.Models.Frame import Cnn_Model_Frame, train
from utils import read_dataset_from_txt
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import math
MODEL_DICT = {
    'SRACN':SRACN,
    'Common_1DCNN': Common_1DCNN,
    'Common_2DCNN': Common_2DCNN,
    'Common_3DCNN': Common_3DCNN,
    "Res_3D_18Net": Res_3D_18Net,
    "Res_3D_34Net": Res_3D_34Net,
    "Res_3D_50Net": Res_3D_50Net,
    'SSRN': SSRN,
    'HybridSN': HybridSN,
    'Vgg16': Vgg16_net,
    'MobileNetV1': MobileNetV1,
    'MobileNetV2': MobileNetV2,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50
}

model_selected = 'Common_1DCNN' # 从上面选择一个模型
config_name = "SRACN-nopre" # 配置输出名称，最后的输出名称为 model_selected_config_name_CurrentTime
out_classes = 11 # 分类数
epochs = 300 # epoch
batch = 48 # batch
init_lr = 3e-4  # lr
min_lr = 3e-6  # 最低学习率
pretrain_pth = r'C:\Users\85002\Downloads\Ete_spectral0_band0.5_noise0.5_Emb128_202508262224_best.pth'
train_images_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\训练集\.datasets.txt'  # 训练数据集
test_images_dir = r'c:\Users\85002\OneDrive\文档\小论文\dataset11classes\d6-4new\测试集\.datasets.txt'  # 测试数据集
ck_pth = None # 用于断点学习
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置
    if_full_cpu = True  # 是否全负荷cpu

    out_embedding = 128
    step_size = epochs // (math.log10(init_lr // min_lr) + 1) # 自动计算学习率调度器的步长
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    print(f'Using num_workers: {dataloader_num_workers}')
    # 配置训练数据集和模型
    train_image_lists = read_dataset_from_txt(train_images_dir) # 使用rewrite好点
    test_image_lists = read_dataset_from_txt(test_images_dir)
    train_dataset = CNN_Dataset(train_image_lists)
    eval_dataset = CNN_Dataset(test_image_lists)
    model = MODEL_DICT[model_selected](out_classes=out_classes, out_embedding=out_embedding, in_shape=train_dataset.data_shape)  # 模型实例化
    print(f"Image shape: {train_dataset.data_shape}")
    if pretrain_pth is not None:
        state_dict = torch.load(pretrain_pth, map_location=device)["backbone"]
        model._load_encoer_params(state_dict) # 加载预训练权重
        model._freeze_encoder() # 冻结编码器参数
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-4)  # 优化器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)  # 学习率调度器
    if step_size <= 0: # step太小,那么不设置调度器
        scheduler = None

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, 
                                  num_workers=dataloader_num_workers, prefetch_factor=2,persistent_workers=True)  # 数据迭代器
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, pin_memory=True, 
                                 num_workers=dataloader_num_workers, prefetch_factor=2,persistent_workers=True)  # 数据迭代器

    frame = Cnn_Model_Frame(model_name=f'{model_selected}_{config_name}', 
                            epochs=epochs, 
                            min_lr=min_lr,
                            device=device, 
                            if_full_cpu=if_full_cpu,)
    
    train(frame=frame,
          model=model, 
          optimizer=optimizer, 
          scheduler=scheduler,
          train_dataloader=train_dataloader, 
          eval_dataloader=eval_dataloader,
          ck_pth=ck_pth)