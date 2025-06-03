import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR,ExponentialLR,ReduceLROnPlateau
from models.Models import CNN_3d
from cnn_model.models.Data import MoniHDF5_leaning_dataset
from tqdm import tqdm
from datetime import datetime
def get_systime():
    return datetime.now().strftime("%Y%m%d%H%M") # 记录系统时间
def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    # Reduces the tensor data across all machines
    # Example: If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt


def setup(rank, world_size):
    # 进程注册
    os.environ['MASTER_ADDR'] = '127.0.0.110'
    os.environ['MASTER_PORT'] = '30000'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def run(rank, world_size):
    epochs = 300
    batch_size = 256
    init_lr = 1e-4
    min_lr = 1e-7
    config_model_name = "CNN_3d"  # 模型名称
    current_script_path = os.getcwd()
    setup(rank, world_size)

    torch.manual_seed(18)
    torch.cuda.manual_seed_all(18)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device(rank)  # 这里设置 device ，后面可以直接使用 data.cuda(),否则需要指定 rank

    # load model
    model = CNN_3d(24, out_classes=8).to(rank)
    optimizer = optim.Adam(model.parameters(), lr=init_lr)  # 优化器
    criterion = nn.CrossEntropyLoss()
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True) # DDP包裹

    dataset_train = 'train_datasets.h5'
    dataset_eval = 'eval_datasets.h5'
    train_dataset = MoniHDF5_leaning_dataset(dataset_train)
    test_dataset = MoniHDF5_leaning_dataset(dataset_eval)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
                               pin_memory=True, num_workers=4, sampler=train_sampler)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              pin_memory=True, num_workers=4, sampler=test_sampler)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

    '''输出模型与日志'''
    if rank == 0:
        current_time = get_systime()
        output_name = config_model_name + '_' + current_time  # 模型输出名称
        log = open(os.path.join(current_script_path, 'logs\\' + output_name + '.log'), 'w')
        model_name = os.path.join(current_script_path, 'models\\' + output_name + ".pth")

    '''训练策略配置'''
    train_epoch_best_accuracy = 0  # 初始化最佳loss
    no_optim = 0  # 用来记录loss不降的轮数
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total_data = 0 # 每个epoch进程训练的总数据量
        total_batch = 0 # 每个epoch进程训练的总次数
        train_sampler.set_epoch(epoch)
        for data, label in tqdm(train_loader, desc='Training:', total=len(train_loader)):
            data, label = data.to(rank).unsqueeze(1), label.to(rank)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            reduced_loss = reduce_tensor(loss.data, world_size) # 平均loss，梯度更新时所有显卡上的参数保持一致
            running_loss += reduced_loss.item()

            _, predict = torch.max(output, 1)
            t_correct = (predict == label).sum()
            reduced_correct = reduce_tensor(t_correct, world_size)  # 全局平均准确
            correct += t_correct.item()
            total_data += data.size(0)
            total_batch += 1
        current_lr = optimizer.param_groups[0]['lr']
        if rank==0:
            avg_loss = running_loss / total_batch
            accuracy = 100 * reduced_correct / total_data
            result = f"Epoch-{epoch + 1} , Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Lr: {current_lr:.8f}"
            log.write(result+'\t')
            print(result)
        dist.barrier() # 阻塞进程
        if test_loader is not None:
            model.eval()
            running_loss = 0.0
            correct = 0
            total_data = 0  # 每个epoch进程训练的总数据量
            total_batch = 0  # 每个epoch进程训练的总次数
            test_sampler.set_epoch(epoch)
            with torch.no_grad():
                for data, label in tqdm(test_loader, desc='Testing', total=len(test_loader)):
                    data, label = data.to(rank).unsqueeze(1), label.to(rank)
                    output = model(data)
                    loss = criterion(output, label)
                    reduced_loss = reduce_tensor(loss.data, world_size)  # 平均loss，梯度更新时所有显卡上的参数保持一致
                    running_loss += reduced_loss.item()

                    _, predict = torch.max(output, 1)
                    t_correct = (predict == label).sum()
                    reduced_correct = reduce_tensor(t_correct, world_size)  # 全局平均准确
                    correct += t_correct.item()
                    total_data += data.size(0)
                    total_batch += 1
                if rank == 0:
                    avg_loss = running_loss / total_batch
                    test_accuracy = 100 * reduced_correct / total_data
                    result = f"Test-Loss: {avg_loss:.4f}, Accuracy: {test_accuracy:.2f}%"
                    log.write(result + '\t')
                    print(result)

        if rank == 0:
            if test_accuracy <= train_epoch_best_accuracy:
                no_optim += 1
            else:  # 若当前epoch的loss小于之前最小的loss
                no_optim = 0  # loss未降低的轮数归0
                train_epoch_best_accuracy = test_accuracy  # 保留当前epoch的accuracy
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(state, model_name)
                print(f"模型参数、优化器参数已保存：{model_name}")
        if (epoch + 1) > 200 or current_lr <= min_lr:
            pass
        else:
            scheduler.step()
    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    # Specify the GPU used
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus
    run_demo(run, world_size)