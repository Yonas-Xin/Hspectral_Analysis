"""针对Train模块的升级版"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
import torch.optim as optim
from cnn_model.Models.Models_New import DATASET_DICT, MODEL_DICT
from torch.optim.lr_scheduler import StepLR
from cnn_model.Models.Frame import Cnn_Model_Frame, train
from utils import read_dataset_from_txt
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import math

if __name__ == '__main__':
    model_name = "Res_3D_18Net" # 使用model_name 与模型库模型匹配
    config_name = "TL" # 配置名称
    out_classes = 8 # 分类数
    epochs = 100 # epoch
    batch = 12 # batch
    init_lr = 1e-3  # lr
    min_lr = 1e-6  # 最低学习率
    GRAGUALLY_UNFRREZE = True
    pretrain_pth = r'C:\Users\85002\Downloads\Moco_Res18_202507242008.pth'
    train_images_dir = r'C:\Users\85002\Desktop\TempDIR\ZY-01-Test\handle_dataset_8classes_400samplesnew\Atrain_datasets.txt'  # 训练数据集
    test_images_dir = r'C:\Users\85002\Desktop\TempDIR\ZY-01-Test\handle_dataset_8classes_400samplesnew\Aeval_datasets.txt'  # 测试数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置
    ck_pth = None # 用于断点学习
    load_from_ck = False  # 从断点处开始训练
    if_full_cpu = True  # 是否全负荷cpu

    out_embeddings = 1024
    # unfreeze_list = [20]
    step_size = epochs // (math.log10(init_lr // min_lr) + 1) # 自动计算学习率调度器的步长
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    print(f'Using num_workers: {dataloader_num_workers}')
    # 配置训练数据集和模型
    train_image_lists = read_dataset_from_txt(train_images_dir) # 使用rewrite好点 
    test_image_lists = read_dataset_from_txt(test_images_dir)
    try:
        train_dataset = DATASET_DICT[model_name](train_image_lists)
        eval_dataset = DATASET_DICT[model_name](test_image_lists)
        model = MODEL_DICT[model_name](out_classes=out_classes, out_embeddings=out_embeddings)  # 模型实例化
    except KeyError as k:
        raise KeyError('model name must be "SARCN", "Shallow_1DCNN" or "Shallow_3DCNN"')
    print(f"Image shape: {train_dataset.data_shape}")
    if pretrain_pth is not None:
        try: 
            state_dict = torch.load(pretrain_pth, map_location=device)["backbone"]
            model._load_encoer_params(state_dict)
            model._freeze_encoder()
            print("预训练权重加载成功！")
        except AttributeError as info:
            GRAGUALLY_UNFRREZE = False
            print(info)
    else: 
        GRAGUALLY_UNFRREZE = False
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=1e-4)  # 优化器
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)  # 学习率调度器
    if step_size <= 0: # step太小,那么不设置调度器
        scheduler = None

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True, pin_memory=True, 
                                  num_workers=dataloader_num_workers, prefetch_factor=2,persistent_workers=True)  # 数据迭代器
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=True, pin_memory=True, 
                                 num_workers=dataloader_num_workers, prefetch_factor=2,persistent_workers=True)  # 数据迭代器

    frame = Cnn_Model_Frame(model_name=f'{model_name}-{config_name}', 
                            epochs=epochs, 
                            min_lr=min_lr,
                            device=device, 
                            if_full_cpu=if_full_cpu,
                            gradually_unfreeze=GRAGUALLY_UNFRREZE)
    
    train(frame=frame,model=model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                train_dataloader=train_dataloader, 
                eval_dataloader=eval_dataloader,
                ck_pth=ck_pth, 
                load_from_ck=load_from_ck)