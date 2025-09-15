import numpy as np
import os
import torch
import matplotlib.colors as mcolors
from datetime import datetime

class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name}: {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_epochs, num_batches, meters, prefix: str = "") -> None:
        self.epoch_fmtstr = self._get_batch_fmtstr(num_epochs)
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch) -> None:
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
    
    def epoch_summary(self, epoch, other_str=""):
        formatted_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entries = ["Epoch" + self.epoch_fmtstr.format(epoch)]
        entries += [str(meter) for meter in self.meters]
        result = f"{formatted_time} "+"\t".join(entries)+"\t"+other_str
        print(result)
        return result

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

def topk_accuracy(output, target, topk=(1,)): # 计算Top-k准确率
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def block_generator(data, block_size=256):
    '''迭代器，输入一个影像，返回分块的位置掩膜'''
    if data.ndim == 3:
        rows, cols, _ = data.shape
    else: rows, cols = data.shape
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # 计算当前块的实际高度和宽度（避免越界）
            actual_rows = min(block_size, rows - i)
            actual_cols = min(block_size, cols - j)
            position_mask = np.zeros((rows, cols))
            position_mask[i:i+actual_rows, j:j+actual_cols] = 1
            yield position_mask
ACADEMIC_COLOR = ['#d5e5c9', '#1c6995', '#d9c2df', '#e2795a', '#eac56c', '#299d90', '#895c56', '#1bb5b9',
                  '#d68e04', '#eea78b', '#459741', '#9566a8', '#a4d2a1', '#e98d49', '#639dfc', '#93a906',]
def label_to_rgb(t, MAP=ACADEMIC_COLOR):
    '''根据颜色条将label映射到rgb图像'''
    MAP = batch_hex_to_rgb(MAP) # 将HEX值转化为rgb值
    if not isinstance(t, (np.int8, np.int16, np.int32, np.int64)):
        t = t.astype(np.int16)
    H, W = t.shape
    t = t.reshape(-1)
    rgb = [ [255, 255, 255] if i == -1 else MAP[i] for i in t ] # -1指定映射为白色
    rgb = np.array(rgb, dtype=np.uint8).reshape(H, W, 3)
    return rgb

def hex_to_rgb(value):
    rgb = mcolors.hex2color(value)
    rgb_255 = [int(round(x * 255)) for x in rgb]
    return rgb_255

def batch_hex_to_rgb(value_list : list):
    return [hex_to_rgb(i) for i in value_list]

def search_files_in_directory(directory, extension):
    """
    搜索指定文件夹中所有指定后缀名的文件，并返回文件路径列表,只适用于不需要标签训练的模型，因为返回的列表顺序可能和
    需要的顺序不同，使用需慎重，但是同一命名规则返回的列表一定是相同的
    Parameters:
        directory (str): 要搜索的文件夹路径
        extension (str): 文件后缀名，应该以 '.' 开头，例如 '.txt', '.jpg'
    Returns:
        list: 包含所有符合条件的文件路径的列表
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files
def read_txt_to_list(filename):
    with open(filename, 'r') as file:
        # 逐行读取文件并去除末尾的换行符
        data = [line.strip() for line in file.readlines()]
    return data

def write_list_to_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # 每个元素后加上换行符
        file.flush()
def read_dataset_from_txt(txt_file):
    'txt文件绝对地址'
    parent_dir = os.path.dirname(txt_file)
    paths = read_txt_to_list(txt_file)
    x = [os.path.basename(i) for i in paths]
    y = [os.path.join(parent_dir, i) for i in x]
    return y

colors = [
    "#C0FDD8", '#FEC0C1', '#CDC6FF', '#FDC0F7', '#F3D8F1',
    '#D6EBBF', '#E1CAF7', '#BFDCE2', '#F8F0BE', '#BEEFBF',
    '#F8C9C8', '#C0E2D2', '#E9BFC0', "#E3E3E3", '#BFBFBF',
    '#DEECF6', '#AFCBE2', '#E2F2CD', '#B6DAA7', '#F9D5D5',
    '#EF9BA1', '#FBE3C0', '#FBC99A', '#EBE0EF', '#C2B1D7',
]

def save_matrix_to_csv(matrix, filename, delimiter=','):
    """
    将 NumPy 矩阵或 PyTorch 张量（二维）保存为 CSV 文件
    
    参数:
        matrix: 输入的 NumPy 矩阵或 PyTorch 张量（二维）
        filename: 要保存的 CSV 文件名（包括路径）
        delimiter: CSV 分隔符，默认为 ','
    
    返回:
        None
    """
    if not isinstance(matrix, (np.ndarray, torch.Tensor)):
        raise ValueError("输入必须是 NumPy 数组或 PyTorch 张量")
    if len(matrix.shape) > 2:
        raise ValueError("输入必须是二维矩阵")
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.detach().cpu().numpy() # 转为 NumPy 数组
    if matrix.dtype == np.bool: # bool 类型转换为 int
        matrix = matrix.astype(np.int16)
    # 使用 NumPy 保存为 CSV
    np.savetxt(filename, matrix, delimiter=delimiter, fmt='%s')
    print(f"data has been saved as csv: {filename}")

def read_csv_to_matrix(filename, delimiter=','):
    """
    从 CSV 文件读取数据并转换为 NumPy 矩阵
    
    参数:
        filename: 要读取的 CSV 文件名（包括路径）
        delimiter: CSV 分隔符，默认为 ','
    
    返回:
        NumPy 矩阵
    """
    return np.loadtxt(filename, delimiter=delimiter, dtype=np.float32)