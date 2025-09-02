"""
根据log文件绘制训练过程中的准确率曲线
"""

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
    # 设置黑色边框
    with plt.rc_context({'axes.edgecolor': 'black',
                        'axes.linewidth': 1.5}):
        ax = plt.gca()
    # plt.style.use('seaborn-v0_8') # 使用seaborn风格

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
    # ax.set_facecolor(ACADEMIC_COLOR[0])  # 设置背景色
    ax.grid(True, 
            linestyle='--', 
            linewidth=0.5, 
            alpha=0.8, 
            color='black')  # 黑色虚线网格
    
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

def plot_accuracy_curves(train_acc_list, val_acc_list, 
                         model_names=None, 
                         title='Training vs Validation Accuracy',
                         save_path=None):
    """
    绘制多个模型的训练集和验证集准确率对比曲线
    
    参数:
        train_acc_list (list): 各模型的训练准确率列表, 如 [model1_train, model2_train, ...]
        val_acc_list (list): 各模型的验证准确率列表, 与train_acc_list顺序一致
        model_names (list): 模型名称列表, 如 ['SARCN', '3D CNN', 'RF']
        title (str): 图表标题
        save_path (str): 图片保存路径
    """
    plt.figure(figsize=(10, 6), dpi=125)
    plt.style.use('seaborn-v0_8')
    
    # 自动生成模型名称
    if model_names is None:
        model_names = [f'Model {i+1}' for i in range(len(train_acc_list))]
    
    # 先绘制所有训练集曲线，再绘制所有验证集曲线
    lines = []  # 用于存储图例句柄
    labels = []  # 用于存储图例标签
    
    # 1. 绘制训练集曲线（实线）
    for i, train_acc in enumerate(train_acc_list):
        color = DEEP_COLOR[i % len(DEEP_COLOR)]
        line = plt.plot(train_acc, 
                       color=color,
                       linewidth=2,
                       linestyle='-',
                       alpha=0.9)
        lines.append(line[0])
        labels.append(f'{model_names[i]} (Train)')
    
    # 2. 绘制验证集曲线（虚线）
    for i, val_acc in enumerate(val_acc_list):
        color = DEEP_COLOR[i % len(DEEP_COLOR)]
        line = plt.plot(val_acc, 
                       color=color,
                       linewidth=2,
                       linestyle='--',
                       alpha=0.9)
        lines.append(line[0])
        labels.append(f'{model_names[i]} (Val)')
    
    # 坐标轴和网格美化
    ax = plt.gca()
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6, color='gray')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  # Epoch为整数
    
    # # 标注最高验证准确率
    # for i, val_acc in enumerate(val_acc_list):
    #     max_idx = np.argmax(val_acc)
    #     plt.scatter(max_idx, val_acc[max_idx], 
    #                color=DEEP_COLOR[i % len(DEEP_COLOR)],
    #                s=80, zorder=5, 
    #                edgecolors='white', linewidths=1.5)
    
    # 标签和标题
    plt.xlabel('Epoch', fontsize=12, labelpad=10)
    plt.ylabel('Accuracy', fontsize=12, labelpad=10)
    plt.title(title, fontsize=14, pad=20)
    
    plt.legend(lines, labels,
               fontsize=10,
               ncol=2,  # 每行显示模型数量
               framealpha=1,
               shadow=True,
               edgecolor='white',)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__=='__main__':
    save_path = 'Comparison.png'
    log_file_path1 = r"C:\Users\85002\Desktop\GF5result\train_process\SSAR_15classes_graunfreeze_nopretrain_202506242120.log"
    train_accuracy1,test_accuracy1 = find_target_from_log(log_file_path1, find_target='Accuracy: ')

    title='Model Accuracy Comparison'
    label = ['Train', 'Val']
    plot_line(train_accuracy1, test_accuracy1, title="Accuracy Curve", labels=label, save_path=save_path)