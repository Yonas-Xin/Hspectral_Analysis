import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import torch.optim as optim
from cnn_model.Models.Models import Constrastive_learning_Model
from cnn_model.Models.Data import Moni_leaning_dataset,MoniHDF5_leaning_dataset
from torch.optim.lr_scheduler import StepLR,ExponentialLR,ReduceLROnPlateau
from cnn_model.Models.Frame import Cnn_model_frame
from utils import read_txt_to_list
from torch.utils.data import DataLoader

if __name__ == '__main__':
    if_full_cpu = True  # 是否全负荷cpu
    load_from_ck = False  # 从断点处开始训练
    epochs = 2  # epoch
    batch = 4 # batch
    init_lr = 1e-4  # lr
    min_lr = 1e-7  # 最低学习率
    warmup_epochs = 0
    pretrain_pth = r'C:\Users\85002\Desktop\模型\Spe_Spa_Attenres110_retrain_202504281258.pth'
    config_model_name = "SSAR"  # 模型名称
    train_images_dir = r'D:\Data\Hgy\research_train_samples\Aeval_datasets.txt'  # 训练数据集
    test_images_dir = r'D:\Data\Hgy\research_train_samples\Aeval_datasets.txt'  # 测试数据集
    ck_pth = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置


    # 配置训练数据集和模型
    train_image_lists = read_txt_to_list(train_images_dir)
    test_image_lists = read_txt_to_list(test_images_dir)
    train_dataset = Moni_leaning_dataset(train_image_lists)
    eval_dataset = Moni_leaning_dataset(test_image_lists)
    model = Constrastive_learning_Model(out_embedding=24, out_classes=8, in_shape=train_dataset.data_shape)  # 模型实例化
    print(f"Image shape: {train_dataset.data_shape}")
    model.load_from_contrastive_model(pretrain_pth, map_location='cuda:0')
    model.freeze_encoder()
    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 学习率调度器

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2,
                            persistent_workers=True)  # 数据迭代器
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2,
                            persistent_workers=True)  # 数据迭代器


    frame = Cnn_model_frame(model_name=config_model_name, 
                            epochs=epochs, 
                            min_lr=min_lr,
                            warmup_epochs=warmup_epochs,
                            device=device, 
                            if_full_cpu=if_full_cpu)
    

    frame.train(model=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                train_dataloader=train_dataloader, 
                eval_dataloader=eval_dataloader,
                ck_pth=ck_pth, 
                load_from_ck=load_from_ck)