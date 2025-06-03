import matplotlib.pyplot as plt
import numpy as np

def find_target_from_log(log_file_path, find_target='Accuracy: '):
    """
    从 .log 文件中提取训练损失数值并返回列表
    """
    train_accuracy = []
    test_accuracy = []
    try:
        with open(log_file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    Accuracy_start = line.find(find_target)
                    if Accuracy_start != -1:
                        loss_str = line[Accuracy_start + len(find_target):].split(',')[0].strip() # strip删除首位空格
                        if '%' in loss_str:
                            loss_str = loss_str[0:-1]
                    try:
                        Accuracy = float(loss_str)
                        if line.startswith("Test"):
                            test_accuracy.append(Accuracy)
                        else:
                            train_accuracy.append(Accuracy)
                    except ValueError:
                        continue
    except FileNotFoundError:
        print(f"错误：文件 {log_file_path} 不存在！")
        return []

    return np.array(train_accuracy), np.array(test_accuracy)

def plot_line(x1,x2):
    '''画图示意'''
    plt.figure(figsize=(12,9))
    plt.plot(x1, label='Train Accuracy')
    plt.plot(x2, label='Test Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__=='__main__':
    log_file_path = r"C:\Users\85002\Desktop\模型\Spe_Spa_Attenres110_retrain_202504281258.log"
    train_accuracy,test_accuracy = find_target_from_log(log_file_path, find_target='Loss: ')
    # log_file_path = r"C:\Users\85002\Desktop\模型\Spe_Spa_Attenres_pretrain_202504252324.log"
    # train_accuracy1,test_accuracy1 = find_target_from_log(log_file_path, find_target='Loss: ')
    # train_accuracy=train_accuracy[:-4]
    # train_accuracy = train_accuracy.tolist()
    # train_accuracy1 = train_accuracy1.tolist()
    # train_accuracy +=train_accuracy1
    # train_accuracy = np.array(train_accuracy)
    # 高斯滤波（标准差=1）
    from scipy.ndimage import gaussian_filter1d

    train_accuracy = gaussian_filter1d(train_accuracy, sigma=1)

    min_loss_idx = np.argmin(train_accuracy)  # 找到最小值位置
    min_loss_value = train_accuracy[min_loss_idx]
    plt.figure(figsize=(8,6))

    plt.scatter(min_loss_idx, min_loss_value, color='red', s=40, label='Min Loss', alpha=1)
    plt.annotate(f'min loss({min_loss_idx}, {min_loss_value:.4f})',
                 xy=(min_loss_idx, min_loss_value),
                 xytext=(min_loss_idx-9, min_loss_value + 0.00001),
                 ha='center', va='bottom',)
    plt.plot(train_accuracy)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(np.arange(0, len(train_accuracy), step=5))  # 调整X轴刻度密度
    # plt.tight_layout()
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    # plt.show()

    plt.savefig('train1.png',dpi=300)
    # plot_line(train_accuracy)