import os
from pathlib import Path
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch
import signal
import shutil
class Cnn_model_frame:
    def __init__(self, model_name, min_lr=1e-7, epochs=300, warmup_epochs=30, device=None, if_full_cpu=True):
        self.loss_func = nn.CrossEntropyLoss()
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
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.parent_dir = os.path.join(base_dir, '_results')  # 创建一个父目录保存训练结果
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
        self.tensorboard_dir = os.path.join(self.parent_dir, f'tensorboard_logs\\logs_{model_save_name}')

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.start_epoch = 0

        signal.signal(signal.SIGINT, self.interrupt_handler)  # 注册中断信号处理函数
        signal.signal(signal.SIGTERM, self.interrupt_handler)  # 注册终止信号处理函数
    def interrupt_handler(self, signum, frame):
        print("\nInterrupt signal received.")
        if not os.path.exists(self.model_path):
            print("Model was not saved. Attempting to clean up log files...")
            self.clean_up()
        exit(1)
    
    def clean_up(self):
        """清理日志文件和tensorboard目录"""
        if os.path.exists(self.log_path):
            self.log_writer.close()  # 确保日志文件被正确关闭
            os.remove(self.log_path)
            print(f"Log file {self.log_path} has been removed.")
        if os.path.exists(self.tensorboard_dir):
            self.writer.close()  # 确保TensorBoard writer被正确关闭
            shutil.rmtree(self.tensorboard_dir)
            print(f"TensorBoard directory {self.tensorboard_dir} has been removed.")

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

    def train(self, model, optimizer, train_dataloader, eval_dataloader=None, scheduler=None, ck_pth=None, load_from_ck=False):
        self.log_writer = open(self.log_path, 'w')
        if not os.path.exists(self.tensorboard_dir):
            os.makedirs(self.tensorboard_dir)
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        model.to(self.device)
        self.load_parameter(model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth, load_from_ck=load_from_ck) # 初始化模型

        try:
            for epoch in range(self.start_epoch, self.epochs):
                model.train()  # 开启训练模式，自训练没有测试模式，所以这个可以在训练之前设置
                running_loss = 0.0
                correct = 0
                total_samples = 0
                for data, label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training:", leave=True):
                    total_samples+=label.shape[0]
                    data, label = data.to(self.device).unsqueeze(1), label.to(self.device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = self.loss_func(output, label)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    _, predict = torch.max(output, 1)
                    correct += (predict == label).sum().item()
                train_avg_loss = running_loss / len(train_dataloader)
                train_accuracy = 100 * correct / total_samples
                current_lr = optimizer.param_groups[0]['lr']
                result = f"Epoch-{epoch + 1} , Loss: {train_avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%, Lr: {current_lr:.8f}"
                self.log_writer.write(result + '\n') # 记录训练过程
                self.writer.add_scalars('Loss', {'Train': train_avg_loss}, epoch)
                self.writer.add_scalars('Accuracy', {'Train': train_accuracy}, epoch)
                print(result)

                if eval_dataloader is not None:
                    model.eval()
                    correct = 0
                    running_loss = 0.0
                    total_samples = 0
                    with torch.no_grad():
                        for data, label in tqdm(eval_dataloader, desc='Testing:', total=len(eval_dataloader), leave=True):
                            total_samples += label.shape[0]
                            data, label = data.to(self.device).unsqueeze(1), label.to(self.device)
                            output = model(data)
                            loss = self.loss_func(output, label)
                            running_loss += loss.item()
                            _, predict = torch.max(output, 1)
                            correct += (predict == label).sum().item()
                        test_avg_loss = running_loss / len(eval_dataloader)
                        test_accuracy = 100 * correct / total_samples
                        result = f"Test_Loss: {test_avg_loss:.4f}, Accuracy: {test_accuracy:.2f}%"
                        self.log_writer.write(result + '\n')  # 记录训练过程
                        self.writer.add_scalars('Loss', {'Test': test_avg_loss}, epoch)
                        self.writer.add_scalars('Accuracy', {'Test': test_accuracy}, epoch)
                        print(result)
                if train_avg_loss <= self.train_epoch_min_loss:
                    self.train_epoch_min_loss = train_avg_loss
                    self.save_model(model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, avg_loss=train_avg_loss)
                if (epoch + 1) < self.warmup_epochs or current_lr <= self.min_lr:
                    pass
                else:
                    scheduler.step()
                self.log_writer.flush()
        except Exception as e:
            self.clean_up()
            print(f"Training interrupted due to: {str(e)}")
            raise
        finally:
            self.log_writer.close() # 再次确保日志文件被正确关闭
            self.writer.close()
            print("Training completed. Program exited.")
            os._exit(0) # 退出主程序