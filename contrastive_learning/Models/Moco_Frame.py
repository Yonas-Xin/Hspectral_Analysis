import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import shutil
from datetime import datetime
import time
import torch.nn as nn
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
import torch
import traceback
# from contrastive_learning.Models.loss import InfoNCELoss
from utils import AverageMeter, ProgressMeter, topk_accuracy

class Moco_Frame:
    '''管理训练参数与训练配置'''
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
        self.model_dir = os.path.join(self.parent_dir, model_save_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, f'{model_save_name}.pth')
        self.model_best_path = os.path.join(self.model_dir, f'{model_save_name}_best.pth')
        self.log_path = os.path.join(self.model_dir, f'{model_save_name}.log')
        self.tensorboard_dir = os.path.join(self.model_dir , f'tensorboard_logs')
        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.train_epoch_min_loss = 100
        self.epoch_max_acc = -1
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

def clean_up(frame):
    """清理因终端或异常生成的文件"""
    if not os.path.exists(frame.model_path):
        if os.path.exists(frame.model_dir):
            shutil.rmtree(frame.model_dir)
            print(f"Directory {frame.model_dir} has been removed.")
    else: pass

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best=False):
    """注意：将需要迁移的部分使用 backbone 存储起来"""
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'best_loss': avg_loss,
        'best_acc': avg_acc,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'current_lr': optimizer.param_groups[0]['lr'],
        'backbone': model.encoder_q.encoder.state_dict(),
    }
    torch.save(state, frame.model_path)
    if is_best:
        shutil.copyfile(frame.model_path, frame.model_best_path)
        print(f"============The best checkpoint saved at epoch {epoch}============")

def load_parameter(frame, model, optimizer, scheduler=None, ck_pth=None, load_from_ck=False): # 加载模型、优化器、调度器
    frame.full_cpu() # 打印配置信息
    if ck_pth is not None:
        if load_from_ck:
            checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)  # 加载断点
            model.load_state_dict(checkpoint['model'])
            frame.train_epoch_min_loss = checkpoint.get('best_loss', 100)
            frame.epoch_max_acc = checkpoint.get('best_acc', -1)
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
    def finish_work():
        try:
            result = f'Model saved at Epoch{model_save_epoch}. \nThe best top1-acc:{frame.epoch_max_acc}. \nThe best training_loss:{frame.train_epoch_min_loss}'
            run_time = cal_time(time.time() - start_time)
            log_writer.write(result + '\n')
            log_writer.write("Program runtime:" + run_time + '\n')
            print(result + '\n' + run_time)
        except:pass
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
    interval_printinfo = max_iter_num // 10 # 每个epoch打印十次过程参数
    loss_note = AverageMeter("Loss", ":.6f")
    top1_acc_note = AverageMeter("Top1-Accuracy", ":.4f")
    top5_acc_note = AverageMeter("Top5-Accuracy", ":.4f") 
    Epoch_wirter = ProgressMeter(frame.epochs, len(dataloader), [loss_note, top1_acc_note, top5_acc_note], "\nStep")
    # start training
    try:
        for epoch in range(frame.start_epoch+1, frame.epochs+1):
            print(f'Epoch {epoch}:')
            for i,block in tqdm(enumerate(dataloader), total=len(dataloader), desc="Train", leave=True):
                block = block.to(frame.device) # batch,C,H,W
                batchs = block.size(0)
                with torch.no_grad():
                    q = frame.augment(block)
                    k = frame.augment(block)
                optimizer.zero_grad()  # 清空梯度
                logits, label = model(q, k)
                loss = frame.loss(logits, label)
                acc1, acc5 = topk_accuracy(logits, label, topk=(1, 5))

                loss_note.update(loss.item(), batchs)
                top1_acc_note.update(acc1.item(), batchs)
                top5_acc_note.update(acc5.item(), batchs)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重
                if (i+1) % interval_printinfo == 0:
                    Epoch_wirter.display(i+1)

            current_lr = optimizer.param_groups[0]['lr']
            epoch_summary = Epoch_wirter.epoch_summary(epoch, f"Lr:{current_lr:.2e}")
            log_writer.write(epoch_summary + '\n') # 记录训练过程
            tensor_writer.add_scalar('Train/Loss', loss_note.avg, epoch) # 记录到tensorboard
            tensor_writer.add_scalar('Train/Top1', top1_acc_note.avg, epoch)
            tensor_writer.add_scalar('Train/Top5', top5_acc_note.avg, epoch)
            is_best = False
            if top1_acc_note.avg > frame.epoch_max_acc: # 使用top1准确率来保存模型
                frame.epoch_max_acc = top1_acc_note.avg
                frame.train_epoch_min_loss = loss_note.avg
                model_save_epoch = epoch
                is_best = True
            elif top1_acc_note.avg == frame.epoch_max_acc:
                if loss_note.avg < frame.train_epoch_min_loss:
                    frame.epoch_max_acc = top1_acc_note.avg
                    frame.train_epoch_min_loss = loss_note.avg
                    model_save_epoch = epoch
                    is_best = True
            save_model(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, 
                       epoch=epoch, avg_loss=loss_note.avg, avg_acc=top1_acc_note.avg, is_best=is_best)
            if current_lr <= frame.min_lr:
                pass
            else:
                if scheduler is not None:
                    scheduler.step()
            loss_note.reset()
            top1_acc_note.reset()
            top5_acc_note.reset()
            log_writer.flush()
        finish_work()
    except KeyboardInterrupt: # 捕获键盘中断信号
        finish_work()
        log_writer.close()
        tensor_writer.close()
        print(f"Training interrupted due to: KeyboardInterrupt")
        clean_up(frame=frame)
    except Exception as e: 
        finish_work()
        log_writer.close()
        tensor_writer.close()
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        clean_up(frame=frame)
    finally:
        log_writer.close()
        tensor_writer.close()
        print(f"Training completed. Program exited.")
        sys.exit(0)

def cal_time(seconds): # 计算时分秒
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    runtime = f'{hours}h {minutes}m {seconds}s'
    return runtime

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