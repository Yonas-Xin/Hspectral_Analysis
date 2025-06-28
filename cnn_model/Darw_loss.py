import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import matplotlib
ACADEMIC_COLOR = ['#d5e5c9', '#d4dee9', '#d9c2df', '#e2795a', '#eac56c', '#299d90', '#895c56', '#1bb5b9',
                  '#d68e04', '#eea78b', '#d5c1d6', '#9566a8', '#a4d2a1', '#e98d49', '#639dfc', '#93a906',]
LINE_COLOR1 = ['#ea272a', '#435aa5', '#6cb48d', '#a47748', '#f7a25c', '#848484']
LINE_COLOR2 = ['#1bb5b9', '#eea78b', '#d5c1d6', '#9566a8', '#a4d2a1', '#e98d49', '#ebcc75', '#489faa']
DEEP_COLOR = ['#e2795a', '#299d90', '#eac56c', '#895c56']
SHALLOW_COLOR = ['#d5e5c9', '#d4dee9', '#d9c2df']
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
def find_target_from_log(log_file_path, find_target='Accuracy: '):
    """
    从 .log 文件中提取匹配值返回训练数据
    """
    if find_target == 'Accuracy: ' or find_target == "Loss: ":
        pass
    else:
        raise ValueError('The find_target must be Accuracy: or Loss:')
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

def plot_line(*args, title='Accuracy Curve', labels=None, save_path=None):
    '''画图示意'''
    plt.figure(figsize=(8, 6), dpi=125)
    plt.style.use('seaborn-v0_8') # 使用seaborn风格

    # 自动生成默认标签
    if labels is None:
        labels = [f'Curve {i+1}' for i in range(len(args))]

    # 绘制所有曲线
    for i, (y_data, label) in enumerate(zip(args, labels)):
        plt.plot(y_data, 
                 label=label,
                 color=DEEP_COLOR[i+1 % len(DEEP_COLOR)],  # 循环使用颜色
                 linewidth=2,
                 alpha=0.9,)

    # 坐标轴和网格美化
    ax = plt.gca()
    # ax.set_facecolor(ACADEMIC_COLOR[0])  # 设置背景色
    ax.grid(True, 
            linestyle='--', 
            linewidth=0.5, 
            alpha=0.6, 
            color='gray')  # 网格线
    
    # 强制x轴为整数（因为epoch是整数）
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  

    # 标签和标题（设置字体和间距）
    plt.xlabel('Epoch', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, labelpad=10)
    if title:
        plt.title(title, fontsize=14, pad=20)

    # 图例美化
    plt.legend(fontsize=12, 
               framealpha=1,      # 去除图例背景透明度
               shadow=True,       # 添加阴影
               edgecolor='white', # 边框颜色
               facecolor=ACADEMIC_COLOR[1],
            #    bbox_to_anchor=(1, 1),  # 将图例移到右侧外部
            #    loc='upper left'
               )  # 图例背景色
    # plt.tight_layout() # 紧密布局
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    save_path = '3D_CNN_202506262008_loss.png'
    log_file_path = r"C:\Users\85002\Desktop\GF5result\train_process\3D_CNN_202506262008.log"
    train_accuracy,test_accuracy = find_target_from_log(log_file_path, find_target='Loss: ')
    label = ['Train', 'Test']
    plot_line(train_accuracy, test_accuracy, title="Accuracy Curve", labels=label, save_path=save_path)