"""
将数据集层次划分为训练集与验证集, 根据txt文件进行
"""

import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import random
import numpy as np
from utils import read_txt_to_list,write_list_to_txt
def split_single_class_to_train_and_eval(dataset:list, ratio, random_seed=None):
    """
    根据给定的比例将两个列表分为训练集和验证集(单个类)
    Parameters:
    data_list (list): 数据列表
    ratio (float): 训练集的比例，默认 0.8
    Returns:
    tuple: (train_data, train_labels, val_data, val_labels)
    """
    # 设置随机种子保证可重复性
    if random_seed is not None:
        random.seed(random_seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    train_size = int(len(dataset) * ratio)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]
    return train_data, val_data,

def split_dataset_to_train_and_eval(dataset:list, ratio):
    """
    根据给定的比例将两个列表分为训练集和验证集(全部数据)，每个类别均已相同的ratio划分
    Parameters:
    data_list (list): 数据列表
    ratio (float): 训练集的比例，默认 0.8
    Returns:
    tuple: (train_data, train_labels, val_data, val_labels)
    """
    label = [data.split()[1] for data in dataset]
    label = np.array(label,dtype=np.int16)
    dataset = np.array(dataset)
    classes = np.unique(label)
    train_data_lists,eval_data_lists = [],[]
    for c in classes:
        mask = (label==c)
        dataset_list = dataset[mask].tolist()
        train_data, eval_data = split_single_class_to_train_and_eval(dataset_list,ratio)
        train_data_lists += train_data
        eval_data_lists += eval_data
    return train_data_lists,eval_data_lists

if __name__ == '__main__':
    dataset_path = r'c:\Users\85002\Desktop\TempDIR\ZY-01-Test\handle_dataset_8classes_400samplesnew\.datasets.txt'

    datasets = read_txt_to_list(dataset_path)
    base_dir = os.path.dirname(dataset_path)

    train_data_lists, eval_data_lists = split_dataset_to_train_and_eval(datasets, 0.2)
    write_list_to_txt(train_data_lists, os.path.join(base_dir, 'train_datasets.txt'))
    write_list_to_txt(eval_data_lists, os.path.join(base_dir, 'eval_datasets.txt'))