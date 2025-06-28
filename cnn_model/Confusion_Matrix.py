import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # 添加进度条
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')  # 使用非GUI后端，彻底绕过tkinter
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['axes.unicode_minus'] = False
def format_value(value):
    percent = value * 100
    if percent == 0:
        return "0.0"
    percent_str = f"{percent:.10f}".rstrip('0').rstrip('.')
    if '.' in percent_str:
        decimals = percent_str.split('.')[-1]
        if len(decimals) > 2:
            return f"{percent:.2f}"
        else:
            return f"{percent:.1f}"
    else:
        return f"{percent_str}"

def plot_confusion_matrix(matrix, 
                          labels=None, 
                          figsize=(10, 8), 
                          cmap='YlOrBr',
                          save_path=None, 
                          percent_mode='precision', 
                          show_colorbar=False, 
                          title = 'Confusion Matrix',
                          label_connector=True,
                          connector_color='gray',
                          connector_style='--'):
    """
    绘制混淆矩阵热力图（支持显示 recall 或 precision 百分比）。

    Parameters:
    - matrix: np.ndarray，混淆矩阵
    - labels: list，分类标签
    - figsize: tuple，图像尺寸
    - cmap: str，颜色映射
    - save_path: str，保存路径
    - percent_mode: None, 'recall', 'precision'，控制显示模式
    - show_colorbar: bool，是否显示 color bar
    """
    plt.figure(figsize=figsize)
    sns.set_theme(font_scale=1.2)
    sns.set_style("white")

    if labels is None:
        labels = [str(i) for i in range(matrix.shape[0])]

    # 百分比计算
    if percent_mode == 'recall':
        norm = matrix.sum(axis=1, keepdims=True)
        matrix_percent = np.divide(matrix, norm, out=np.zeros_like(matrix, dtype=float), where=norm != 0)
        annot = np.vectorize(format_value)(matrix_percent)
        data = matrix_percent
        fmt = ''
    elif percent_mode == 'precision':
        norm = matrix.sum(axis=0, keepdims=True)
        matrix_percent = np.divide(matrix, norm, out=np.zeros_like(matrix, dtype=float), where=norm != 0)
        annot = np.vectorize(format_value)(matrix_percent)
        data = matrix_percent
        fmt = ''
    else:
        annot = matrix
        data = matrix
        fmt = 'd'

    ax = sns.heatmap(data,
                     annot=annot,
                     fmt=fmt,
                     cmap=cmap,
                     xticklabels=labels,
                     yticklabels=labels,
                     linewidths=0,
                     linecolor='white',
                     cbar=show_colorbar,
                     cbar_kws={'shrink': 0.8, 'aspect': 20} if show_colorbar else None)
    
    if title is not None:
        ax.set_xlabel('Predicted Label', fontsize=14)
        ax.set_ylabel('True Label', fontsize=14)
        ax.set_title(title, fontsize=16, pad=12)

    ax.set_xticklabels(labels, 
                      rotation=0, 
                      ha='center',
                      va='top')     # 顶端对齐
    
    ax.set_yticklabels(labels,
                      rotation=0,
                      ha='right',   # 右侧对齐
                      va='center')  # 垂直居中

    rows, cols = matrix.shape
    ax.add_patch(plt.Rectangle((0, 0), cols, rows,
                               fill=False, edgecolor='black', linewidth=2, clip_on=False))
    # 为标签和坐标轴添加连接线
    # 添加短连接线（标签对齐）
    for i in range(len(labels)):
        # x轴标签向下的短线
        x = i + 0.5
        y_bottom = ax.get_ylim()[0]
        ax.plot([x, x], [y_bottom, y_bottom + 0.185], color='black', linewidth=1.0, clip_on=False)

        # y轴标签向左的短线
        y = i + 0.5
        x_left = ax.get_xlim()[0]
        ax.plot([x_left - 0.15, x_left], [y, y], color='black', linewidth=1.0, clip_on=False)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    save_path = '3DCNN.png'
    percent_mode = 'precision'
    title = None

    matrix = np.array([[39,  0,  3,  0,  2,  0,  1,  0,  2,  0,  3,  1,  0,  0,  0],
 [ 1, 19,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 5,  0, 11,  0,  0,  0,  0,  0,  2,  0,  0,  2,  0,  0,  0],
 [ 0,  0,  0, 17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
 [ 3,  0,  0,  0, 20,  0,  0,  0,  5,  0,  0,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  0, 37,  1,  0,  0,  0,  0,  2,  0,  0,  2],
 [ 0,  0,  1,  0,  1,  0, 22,  0,  1,  0,  0,  1,  0,  0,  0],
 [ 0,  0,  3,  0,  0,  0,  0, 17,  3,  0,  1,  2,  0,  0,  0],
 [ 1,  0,  1,  0,  4,  0,  1,  5, 38,  2,  3,  0,  0,  0,  0],
 [ 0,  0,  0,  0,  1,  0,  0,  2,  2, 30,  1,  0,  0,  0,  0],
 [ 2,  0,  0,  0,  0,  0,  0,  4,  0,  1, 35,  2,  3,  0,  0],
 [ 3,  0,  3,  0,  0,  0,  0,  0,  0,  0,  0, 33,  1,  0,  3],
 [ 0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  3,  0, 31,  4,  0],
 [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, 21,  0],
 [ 1,  0,  3,  0,  0,  1,  2,  0,  0,  0,  0,  4,  0,  0, 12]])
    plot_confusion_matrix(matrix=matrix, 
                          labels=None, 
                          figsize=(10, 8), 
                          cmap='YlOrBr',
                          save_path=save_path, 
                          percent_mode=percent_mode, 
                          show_colorbar=False, 
                          title = title,)