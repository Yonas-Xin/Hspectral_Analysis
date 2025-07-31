import os
import sys
import torch.nn as nn
import numpy as np
from datetime import datetime
from tqdm import tqdm
from multiprocessing import cpu_count
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import torch
import shutil
from utils import AverageMeter, ProgressMeter, topk_accuracy
import traceback

class Cnn_Model_Frame:
    def __init__(self, model_name, min_lr=1e-7, epochs=300, device=None, if_full_cpu=True, gradually_unfreeze=True):
        self.loss_func = nn.CrossEntropyLoss()
        self.min_lr = min_lr
        self.epochs = epochs
        self.GRAGUALLY_UNFRREZE = gradually_unfreeze

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
        self.model_dir = os.path.join(self.parent_dir, model_save_name)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        self.model_path = os.path.join(self.model_dir, f'{model_save_name}.pth')
        self.model_best_path = os.path.join(self.model_dir, f'{model_save_name}_best.pth')
        self.log_path = os.path.join(self.model_dir, f'{model_save_name}.log')
        self.tensorboard_dir = os.path.join(self.model_dir , f'tensorboard_logs')

        #配置训练信息
        self.if_full_cpu = if_full_cpu
        self.test_epoch_min_loss = 100
        self.test_epoch_max_acc = -1
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

def load_parameter(frame, model, optimizer, scheduler=None, ck_pth=None, load_from_ck=False): # 加载模型、优化器、调度器
    frame.full_cpu() # 打印配置信息
    if ck_pth is not None:
        if load_from_ck:
            checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)
            model.load_state_dict(checkpoint['model'])
            frame.test_epoch_min_loss = checkpoint.get('best_loss', 100)
            frame.test_epoch_max_acc = checkpoint.get('best_acc', -1)
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

def save_model(frame, model, optimizer, scheduler, epoch=None, avg_loss=None, avg_acc=None, is_best = False):
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
    if is_best:
        shutil.copyfile(frame.model_path, frame.model_best_path)
        print(f"============The best checkpoint saved at epoch {epoch + 1}============")

def train(frame, model, optimizer, train_dataloader, eval_dataloader=None, scheduler=None, ck_pth=None, load_from_ck=False):
    log_writer = open(frame.log_path, 'w')
    if not os.path.exists(frame.tensorboard_dir):
        os.makedirs(frame.tensorboard_dir)
    tensor_writer = SummaryWriter(log_dir=frame.tensorboard_dir)
    model.to(frame.device)
    load_parameter(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, ck_pth=ck_pth, load_from_ck=load_from_ck) # 初始化模型

    best_train_accuracy = 0
    best_test_accuracy = 0
    model_save_epoch = 0
    train_loss_note = AverageMeter("Train-Loss", ":.6f")
    train_acc_note = AverageMeter("Train-Acc", ":.4f")
    test_loss_note = AverageMeter("Test-Loss", ":.6f")
    test_acc_note = AverageMeter("Test-Acc", ":.4f")
    progress_writer = ProgressMeter(frame.epochs, len(train_dataloader), 
                                    [train_loss_note, train_acc_note, test_loss_note, test_acc_note],
                                    prefix="Batch")
    try:
        for epoch in range(frame.start_epoch, frame.epochs):
            model.train()  # 开启训练模式，自训练没有测试模式，所以这个可以在训练之前设置
            if frame.GRAGUALLY_UNFRREZE: # 添加逐级解冻策略
                try:
                    model.gradually_unfreeze_encoder_modules(epoch)
                except Exception as e:
                    print(f'The parameter unfreezing was unsuccessful:{e}')
                    pass
            for data, label in tqdm(train_dataloader, total=len(train_dataloader), desc="Training:", leave=True):
                batchs = data.size(0)
                data, label = data.to(frame.device), label.to(frame.device)
                optimizer.zero_grad()
                output = model(data)
                loss = frame.loss_func(output, label)
                acc = topk_accuracy(output, label)

                train_loss_note.update(loss.item(), batchs)
                train_acc_note.update(acc[0].item(), batchs)
                loss.backward()
                optimizer.step()

            if eval_dataloader is not None:
                model.eval()
                with torch.no_grad():
                    for data, label in tqdm(eval_dataloader, desc='Testing:', total=len(eval_dataloader), leave=True):
                        batchs = data.size(0)
                        data, label = data.to(frame.device), label.to(frame.device)
                        output = model(data)
                        loss = frame.loss_func(output, label)
                        acc = topk_accuracy(output, label)
                        test_loss_note.update(loss.item(), batchs)
                        test_acc_note.update(acc[0].item(), batchs)

            test_accuracy = test_acc_note.avg
            test_avg_loss = test_loss_note.avg
            train_accuracy = train_acc_note.avg
            train_avg_loss = train_loss_note.avg

            current_lr = optimizer.param_groups[0]['lr']
            if current_lr <= frame.min_lr:
                pass
            else:
                if scheduler is not None:
                    scheduler.step()
            result = progress_writer.epoch_summary(epoch+1, f"Lr: {current_lr:.2e}")
            tensor_writer.add_scalars('Loss', {'Train': train_avg_loss}, epoch)
            tensor_writer.add_scalars('Accuracy', {'Train': train_accuracy}, epoch)
            tensor_writer.add_scalars('Loss', {'Test': test_avg_loss}, epoch)
            tensor_writer.add_scalars('Accuracy', {'Test': test_accuracy}, epoch)
            log_writer.write(result + "\n")
            log_writer.flush()
            train_loss_note.reset()
            train_acc_note.reset()
            test_loss_note.reset()
            test_acc_note.reset()
            is_best = False
            if test_accuracy > frame.test_epoch_max_acc: # 使用测试集的预测准确率进行模型保存
                frame.test_epoch_min_loss = test_avg_loss
                frame.test_epoch_max_acc = test_accuracy
                best_train_accuracy = train_accuracy
                best_test_accuracy = test_accuracy
                model_save_epoch = epoch
                is_best = True
            elif test_accuracy == frame.test_epoch_max_acc:
                if test_avg_loss < frame.test_epoch_min_loss:
                    frame.test_epoch_min_loss = test_avg_loss
                    frame.test_epoch_max_acc = test_accuracy
                    best_train_accuracy = train_accuracy
                    best_test_accuracy = test_accuracy
                    model_save_epoch = epoch
                    is_best = True
            save_model(frame=frame, model=model, optimizer=optimizer, scheduler=scheduler, epoch=epoch, 
                       avg_loss=test_avg_loss, avg_acc=test_accuracy, is_best=is_best)


        best_result = f'Model saved at Epoch{model_save_epoch}. \nThe best training_acc:{best_train_accuracy:.6f}%.\nThe best testing_acc:{best_test_accuracy:.6f}%'
        log_writer.write(best_result + '\n')
        print(best_result)
        if eval_dataloader is not None:
            print_result_report(frame=frame, model=model, ck_pth=frame.model_best_path, eval_dataloader=eval_dataloader, log_writer=log_writer) # 训练完成打印报告
    except KeyboardInterrupt: # 捕获键盘中断信号
        log_writer.close()
        tensor_writer.close()
        print(f"Training interrupted due to: KeyboardInterrupt")
        clean_up(frame=frame)
    except Exception as e: 
        log_writer.close()
        tensor_writer.close()
        print(traceback.format_exc())  # 打印完整的堆栈跟踪
        clean_up(frame=frame)
    finally:
        log_writer.close() # 再次确保日志文件被正确关闭
        tensor_writer.close()
        sys.exit(0)
    
def print_result_report(frame, model, ck_pth, eval_dataloader, log_writer):
    checkpoint = torch.load(ck_pth, weights_only=True, map_location=frame.device)
    model.load_state_dict(checkpoint['model']) # 从保存的最佳模型中加载参数
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, label in tqdm(eval_dataloader, desc='Generating Report', total=len(eval_dataloader)):
            data, label = data.to(frame.device), label.to(frame.device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        clf_report = classification_report(all_labels, all_preds, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        log_writer.write(f"\n\nTest_acc: {accuracy:.4f}\n")
        log_writer.write(f"Cohen's Kappa: {kappa:.4f}")
        log_writer.write(f"Classification Report:\n")
        log_writer.write(clf_report + "\n\n")
        log_writer.write("Confusion Matrix:\n")
        log_writer.write(np.array2string(conf_matrix, separator=', '))
        log_writer.write('\n')
        log_writer.flush()

        print("Test Accuracy:", accuracy)
        print(f"Cohen's Kappa: {kappa:.4f}")
        print("Classification Report:\n", clf_report)
        print("Confusion Matrix:\n", conf_matrix)