import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import torch
from cnn_model.Models.Models import MODEL_DICT, DATASET_DICT
from utils import rewrite_paths_info
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
    writer_name = '1DCNN.log'
    saved_model_name = r'D:\Programing\pythonProject\Hspectral_Analysis\cnn_model\_results\models_pth\SSAR_202506261846.pth'
    batch = 36 # batch
    test_images_dir = r'd:\Data\Hgy\龚鑫涛试验数据\program_data\handle_class\clip_data_15classs_1x1\Aeval_datasets.txt'  # 测试数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 显卡设置
    out_classes = 15 # 分类数


    out_embeddings = 24 # 模型初始化必要，后面打算把这个参数设置为固定值
    print(f'Using num_workers: {dataloader_num_workers}')
    # 配置训练数据集和模型
    test_image_lists = rewrite_paths_info(test_images_dir)
    eval_dataset = DATASET_DICT[model_name](test_image_lists)
    model = MODEL_DICT[model_name](out_embedding=out_embeddings, out_classes=out_classes, in_shape=eval_dataset.data_shape)  # 模型实例化
    model.load_state_dict(torch.load(saved_model_name, weights_only=True, map_location=device)['model'])
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch, shuffle=False, pin_memory=True, num_workers=0)  # 数据迭代器
    log_writer = open(writer_name, 'w')
    print_result_report(model=model, eval_dataloader=eval_dataloader, log_writer=log_writer, device=device)