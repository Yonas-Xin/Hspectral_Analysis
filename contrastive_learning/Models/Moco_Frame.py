import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import signal
import shutil
from pathlib import Path
from datetime import datetime
import time
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch
import traceback
# from contrastive_learning.Models.loss import InfoNCELoss
from utils import save_matrix_to_csv

class Moco_Frame:
    '''配置训练参数与模型保存地址等'''
    def __init__(self, augment, model_name, min_lr=1e-7, epochs=300, device=None, if_full_cpu=True):
        self.augment = augment
        self.loss = nn.CrossEntropyLoss()
        self.min_lr = min_lr
        self.epochs=epochs

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device

        # 配置输出模型的名称和日志名称
        current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
        model_save_name = f'{model_name}_{current_time}'
        self.parent_dir = os.path.join(base_path, '_results') # 创建一个父目录保存训练结果
        if not os.path.exists(self.parent_dir):
            os.makedirs(self.parent_dir)
        model_dir = os.path.join(self.parent_dir, 'models_pth')
        log_dir = os.path.join(self.parent_dir, 'logs')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.model_path = os.path.join(model_dir, f'{model_save_name}.pth')
        self.log_path = os.path.join(log_dir, f'{model_save_name}.log')
        self.tensorboard_dir = os.path.join(self.parent_dir, f'tensorboard_logs\\{model_save_name}')

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.start_epoch = 0

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
        print(f'Cuda device count: {torch.cuda.device_count()} And the current device:{self.device}')  # 显卡数

def clean_up(frame,log_writer,tensor_writer):
    """清理日志文件和tensorboard目录"""
    if not os.path.exists(frame.model_path):
        if os.path.exists(frame.log_path):
            log_writer.close()
            os.remove(frame.log_path)
            print(f"Removed log file: {frame.log_path}")
        if os.path.exists(frame.tensorboard_dir):
            tensor_writer.close()
            shutil.rmtree(frame.tensorboard_dir)
            print(f"Removed tensorboard directory: {frame.tensorboard_dir}")
        else: pass

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': avg_loss,
        'best_acc': avg_acc,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr']
    }
    torch.save(state, frame.model_path)
    print(f"============Checkpoint saved at epoch {epoch + 1}============")

def load_parameter(frame, model, optimizer, scheduler=None, ck_pth=None, load_from_ck=False): # 加载模型、优化器、调度器
    frame.full_cpu() # 打印配置信息
    if ck_pth is not None:
        if load_from_ck:
            checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)  # 加载断点
            model.load_state_dict(checkpoint['model'])
            frame.train_epoch_min_loss = checkpoint.get('best_loss', 100)
            try:
                optimizer.load_state_dict(checkpoint['optimizer']) # 恢复优化器
                print('The optimizer state have been loaded!')
                frame.start_epoch = checkpoint.get('epoch', -1) + 1  # 获取epoch信息，如果没有，默认为0
            except(ValueError, RuntimeError):
                print('The optimizer is incompatible, and the parameters do not match')
            if scheduler and 'scheduler' in checkpoint: # 恢复调度器
                try:
                    scheduler.load_state_dict(checkpoint['scheduler'])
                except (ValueError, RuntimeError):
                    print('The scheduler is incompatible')
            print(f"Loaded checkpoint from epoch {frame.start_epoch}, current lr {optimizer.param_groups[0]['lr']}")

def train(frame:Moco_Frame, model, optimizer, dataloader, scheduler=None, ck_pth=None, load_from_ck=False):
    start_time = time.time()
    log_writer = open(frame.log_path, 'w')
    if not os.path.exists(frame.tensorboard_dir):
        os.makedirs(frame.tensorboard_dir)
    tensor_writer = SummaryWriter(log_dir=frame.tensorboard_dir)
    model.to(frame.device)
    load_parameter(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth, load_from_ck=load_from_ck) # 初始化模型
    model.train() # 开启训练模式，自训练没有测试模式，所以这个可以在训练之前设置
    
    model_save_epoch = 0
    max_iter_num = len(dataloader)
    interval_printinfo = max_iter_num // 10 # 隔10次打印一次loss

    # start training
    try:
        for epoch in range(frame.start_epoch, frame.epochs):
            running_loss = 0.0
            for i,block in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=True):
                block = block.to(frame.device) # batch,C,H,W
                with torch.no_grad():
                    q = frame.augment(block)
                    k = frame.augment(block)
                optimizer.zero_grad()  # 清空梯度
                logits, label = model(q, k)

                loss = frame.loss(logits, label)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                running_loss += loss.item()
                if (i+1) % interval_printinfo == 0:
                    print(f"\nStep: {i}, The current loss: {loss:.6f}, The Avgloss: {running_loss/(i+1):.6f}")
            avg_loss = running_loss / len(dataloader)
            current_lr = optimizer.param_groups[0]['lr']
            result = f"Epoch-{epoch + 1} , Loss: {avg_loss:.8f}, Lr: {current_lr:.8f}"
            log_writer.write(result + '\n') # 记录训练过程
            tensor_writer.add_scalar('Train/Loss', avg_loss, epoch) # 记录到tensorboard
            print(result)
            if avg_loss >= frame.train_epoch_min_loss:  # 若当前epoch的loss大于等于之前最小的loss
                pass
            else:
                frame.train_epoch_min_loss = avg_loss
                model_save_epoch = epoch
                save_model(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, avg_loss=avg_loss)
            if current_lr <= frame.min_lr:
                pass
            else:
                if scheduler is not None:
                    scheduler.step()
            log_writer.flush()
        
        # 打印和记录结果
        result = f'Model saved at Epoch{model_save_epoch}. \nThe best training_loss:{frame.train_epoch_min_loss}'
        end_time = time.time()
        total_seconds = end_time - start_time
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        runtime = f'Program runtime: {hours}h {minutes}m {seconds}s'
        log_writer.write(result + '\n')
        log_writer.write(runtime + '\n')
        print(result)
        print(runtime)
    except KeyboardInterrupt: # 捕获键盘中断信号
        print(f"Training interrupted due to: KeyboardInterrupt")
        clean_up(frame=frame, log_writer=log_writer, tensor_writer=tensor_writer)
    except Exception as e: 
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        clean_up(frame=frame, log_writer=log_writer, tensor_writer=tensor_writer)
    finally:
        log_writer.close()
        tensor_writer.close()
        print(f"Training completed. Program exited.")
        sys.exit(0)


class Contrasive_learning_predict_frame:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else: self.device = device
        self.out_embedding = None
    
    def predict(self, model, dataloader):
        model.to(self.device)
        with torch.no_grad():
            model.eval()
            idx = 0
            for image in tqdm(dataloader, total=len(dataloader)):
                image = image.unsqueeze(1).to(self.device)
                predict = model.predict(image)
                if self.out_embedding is None:
                    # 初始化输出嵌入矩阵，预分配内存
                    embedding_nums = predict.shape[-1]
                    self.out_embedding = torch.empty((len(dataloader.dataset), embedding_nums), dtype=torch.float32, device=self.device)
                self.out_embedding[idx:idx+len(predict)] = predict
                idx += len(predict)
        self.out_embedding = self.out_embedding.cpu().numpy()
        return self.out_embedding
