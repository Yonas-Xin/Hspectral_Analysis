import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from pathlib import Path
from datetime import datetime
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch
from models.loss import InfoNCELoss
class Cl_Frame:
    def __init__(self, augment, model_name, min_lr=1e-7, epochs=300, warmup_epochs=30,device=None, if_full_cpu=True):
        self.augment = augment
        self.infonce = InfoNCELoss(temperature=0.07)
        self.min_lr = min_lr
        self.epochs=epochs
        if warmup_epochs is not None:
            self.warmup_epochs = warmup_epochs
        else: self.warmup_epochs = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        # 配置输出模型的名称和日志名称
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        model_save_name = f'{model_name}_{current_time}'
        self.parent_dir = Path(__file__).parent # 获取当前运行脚本的目录
        model_dir = os.path.join(self.parent_dir, 'models')
        log_dir = os.path.join(self.parent_dir, 'logs')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_path = os.path.join(model_dir, f'{model_save_name}.pth')
        self.log_path = os.path.join(log_dir, f'{model_save_name}.log')
        self.tensorboard_dir = os.path.join(self.parent_dir, f'tensorboard_logs\\logs_{model_save_name}')

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.start_epoch = 0
        self.del_logs() # 删除多余logs

    def load_parameter(self, model, optimizer, scheduler=None, ck_pth=None, load_from_ck=False): # 加载模型、优化器、调度器
        self.full_cpu() # 打印配置信息
        if ck_pth is not None:
            checkpoint = torch.load(ck_pth, weights_only=True)
            model.load_state_dict(checkpoint['model'])
            if load_from_ck:
                self.train_epoch_min_loss = checkpoint.get('best_loss', 100)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer']) # 恢复优化器
                    print('The optimizer state have been loaded!')
                    self.start_epoch = checkpoint.get('epoch', -1) + 1  # 获取epoch信息，如果没有，默认为0
                except(ValueError, RuntimeError):
                    print('The optimizer is incompatible, and the parameters do not match')
                if scheduler and 'scheduler' in checkpoint: # 恢复调度器
                    try:
                        scheduler.load_state_dict(checkpoint['scheduler'])
                    except (ValueError, RuntimeError):
                        print('The scheduler is incompatible')
            print(f"Loaded checkpoint from epoch {self.start_epoch}, current lr {optimizer.param_groups[0]['lr']}")

    def full_cpu(self):
        cpu_num = cpu_count()  # 自动获取最大核心数目
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        if self.if_full_cpu:
            torch.set_num_threads(cpu_num)
            print('Using cpu core num: ', cpu_num)
        print('Is cuda availabel: ', torch.cuda.is_available())  # 是否支持cuda
        print('Cuda device count: ', torch.cuda.device_count())  # 显卡数
        print('Current device: ', torch.cuda.current_device())  # 当前计算的显卡id

    def save_model(self, model, optimizer, scheduler, epoch=None, avg_loss=None):
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'best_loss': avg_loss,
            'scheduler': scheduler.state_dict() if scheduler else None,
            'current_lr': optimizer.param_groups[0]['lr']
        }
        torch.save(state, self.model_path)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def train(self, model, optimizer, dataloader, scheduler=None, ck_pth=None, clean_noise_samples=False, load_from_ck=False, clean_th=0.99):
        log_writer = open(self.log_path, 'w')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        writer = SummaryWriter(log_dir=self.tensorboard_dir)
        model.to(self.device)
        self.load_parameter(model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth, load_from_ck=load_from_ck) # 初始化模型
        model.train() # 开启训练模式，自训练没有测试模式，所以这个可以在训练之前设置
        for epoch in range(self.start_epoch, self.epochs):
            running_loss = 0.0
            for block in tqdm(dataloader, total=len(dataloader), desc="Train", leave=True):
                block = block.to(self.device) # batch,C,H,W
                block1 = self.augment(block).detach()
                block2 = self.augment(block).detach()
                optimizer.zero_grad()  # 清空梯度
                embedding, out = model(torch.cat((block1, block2), dim=0).unsqueeze(1))
                if clean_noise_samples:
                    self.infonce.cosine_similarity_matrix(embedding, th=clean_th)

                loss = self.infonce(out)  # 对比损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                running_loss += loss.item()
            avg_loss = running_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            result = f"Epoch-{epoch + 1} , Loss: {avg_loss:.8f}, Lr: {current_lr:.8f}"
            log_writer.write(result + '\n') # 记录训练过程
            writer.add_scalar('Train/Loss', avg_loss, epoch) # 记录到tensorboard
            print(result)
            if avg_loss >= self.train_epoch_min_loss:  # 若当前epoch的loss大于等于之前最小的loss
                pass
            else:
                self.train_epoch_min_loss = avg_loss
                self.save_model(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, avg_loss=avg_loss)
            if (epoch + 1) < self.warmup_epochs or current_lr <= self.min_lr:
                pass
            else:
                scheduler.step()
            log_writer.flush()
        log_writer.close()
        writer.close()

    def del_logs(self):
        '''删除多余logs文件'''
        get_basename = lambda f: os.path.splitext(f)[0]
        logs_dir = os.path.join(self.parent_dir, 'logs')
        models_dir = os.path.join(self.parent_dir, 'models')
        log_files = {get_basename(f) for f in os.listdir(logs_dir)
                     if os.path.isfile(os.path.join(logs_dir, f))}
        model_files = {get_basename(f) for f in os.listdir(models_dir)
                       if os.path.isfile(os.path.join(models_dir, f))}
        for f in os.listdir(logs_dir):
            if (get_basename(f) in log_files - model_files and
                    os.path.isfile(os.path.join(logs_dir, f))):
                os.remove(os.path.join(logs_dir, f))
        print("Excess log files cleaned up")