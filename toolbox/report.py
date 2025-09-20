"""
根据保存的模型，对样本集进程评测，生成报告
"""


import sys, os
sys.path.append('.')
import torch
from cnn_model.Models.Data import CNN_Dataset
from utils import read_dataset_from_txt
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
import numpy as np

def print_result_report(model, eval_dataloader, log_writer, device):
    model.to(device)
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for data, label in tqdm(eval_dataloader, desc='Generating Report', total=len(eval_dataloader)):
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, preds = torch.max(output, 1)
            all_labels.extend(label.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        clf_report = classification_report(all_labels, all_preds, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        log_writer.write(f"\n\nTest_acc: {accuracy:.4f}\n")
        log_writer.write(f"Cohen's Kappa: {kappa:.4f}\n")
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
if __name__ == '__main__':
    model_name = "Shallow_1DCNN"
    log_path = '1DCNN.log'
    model_pth = r'D:\Programing\pythonProject\Hspectral_Analysis\cnn_model\_results\models_pth\SSAR_202506261846.pth'
    batch = 36 # batch
    test_images_dir = r'D:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\clip_test_dataset_1x1\.datasets.txt'  # 测试数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置
    out_classes = 15 # 分类数


    out_embeddings = 128 # 模型初始化必要，后面打算把这个参数设置为固定值
    # 配置训练数据集和模型
    test_image_lists = read_dataset_from_txt(test_images_dir)
    eval_dataset = CNN_Dataset(test_image_lists)
    model = torch.load(model_pth, weights_only=False, map_location=device)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=0)  # 数据迭代器
    log_writer = open(log_path, 'w')
    print_result_report(model=model, eval_dataloader=eval_dataloader, log_writer=log_writer, device=device)