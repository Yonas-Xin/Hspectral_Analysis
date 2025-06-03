import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm  # 添加进度条
from cnn_model.models.Data import Moni_leaning_dataset,MoniHDF5_leaning_dataset
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader
from models.Encoder import ECA_SpectralAttention_3d,common_3d,space_speactral_3d,nn,F
import torch
from models.Decoder import deep_classfier
import matplotlib
from models.Models import CNN_3d

matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False
class Spa_Spe_25x25(nn.Module):
    '''加入光谱注意力机制，不加入空间注意力'''
    def __init__(self,out_embedding=20):
        super().__init__()
        self.spectral_attention = ECA_SpectralAttention_3d(138, 2,1) # 全局平均池化
        self.conv_block1 = common_3d(1,64,(5,1,1),(2,0,0), 1)
        self.conv_block2 = space_speactral_3d(64,128,(3,3,3),(1,1,1),1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv_block3 = space_speactral_3d(128,256,(3,3,3),(1,1,1),1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv_block4 = space_speactral_3d(256,256,(3,3,3),(1,1,1),1)
        self.pool3 = nn.MaxPool3d(2)
        self.linear = nn.Linear(39168, out_features=out_embedding)
        self.dp = nn.Dropout(p=0.25) # 设置一个dropout层
    def forward(self, x):
        x = self.conv_block1(self.spectral_attention(x))
        x = self.pool1(self.conv_block2(x))
        x = self.pool2(self.conv_block3(x))
        x = self.pool3(self.conv_block4(x))
        x = x.view(x.shape[0], -1)
        return self.linear(self.dp(x))
class classfier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, 128)
        self.fc2 = nn.Linear(128, 4096)
        self.fc3 = nn.Linear(4096, out_channels)
        self.dp = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x),inplace=True)
        x = F.relu(self.fc2(x),inplace=True) # 输出语义
        x = self.dp(x)
        return self.fc3(x)

class Model(nn.Module):
    def __init__(self, out_embedding=10, out_classes=8):
        super().__init__()
        self.encoder = Spa_Spe_25x25(out_embedding=out_embedding)
        self.decoder = classfier(out_embedding, out_classes)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(F.relu(x, inplace=True))
        return x

def compute_confusion_matrix(model, Dataloer, label_nums):
    '''返回真实标签和预测标签'''
    model.eval()
    model.cuda()
    true_labels = torch.empty((label_nums,), dtype=torch.int16).cuda()
    all_preds = torch.empty((label_nums,), dtype=torch.int16).cuda()
    idx = 0
    with torch.no_grad():
        for inputs, labels in tqdm(Dataloer,total=len(Dataloer)):
            batch = inputs.size(0)
            inputs = inputs.unsqueeze(1).cuda()
            labels = labels.cuda()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            true_labels[idx:idx+batch,] = labels
            all_preds[idx:idx+batch,] = preds
            idx += batch
    all_labels = true_labels.unsqueeze(1).cpu().numpy()
    all_preds = all_preds.unsqueeze(1).cpu().numpy()
    return all_labels, all_preds

def draw_confusion_matrix(true_labels=None, pred_labels=None, cm=None, if_norm=True,
                          class_name=None, out_name='Confusion_matrix.npz', out_figname='混淆矩阵.png'):
    if true_labels is not None:
        cm = confusion_matrix(true_labels, pred_labels)
        if os.path.exists(out_name):
            print('Confusion_Matrix 已存在，文件不覆盖')
        else:
            np.savez_compressed(out_name, cm)
        kappa = cohen_kappa_score(true_labels, pred_labels)
        # 打印kappa系数
        print(f"Cohen's Kappa系数: {kappa:.3f}")
        # 打印分类报告
        print("Classification Report:")
        print(classification_report(true_labels, pred_labels))
    if class_name is None:
        class_name = np.arange(cm.shape[0])
    if if_norm: # 如果归一化，则矩阵的值变为precision，即精度
        cm = cm.astype(np.float32) / cm.sum(axis=0, keepdims=True)
        cm*=100 # 变为百分数

    plt.figure(figsize=(12,9))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_name,
                yticklabels=class_name)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    # plt.xticks(rotation=45, ha='right') # x轴标签旋转45度
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_figname, dpi=450)
    plt.show()

class_names = [
    '冰雪覆盖物',
    '晚更新世洪冲积物',
    '金水口群大理岩',
    '鄂拉山组火山碎屑岩',
    '中更新世冰水堆积物',
    '全新世冲积物',
    '二长花岗岩',
    '正长花岗岩'
]
if __name__ == '__main__':
    Confusion_matrix_name = 'Confusion_Matrix.npz'
    model = Model(24,8)
    state = torch.load('D:\Programing\pythonProject\Hyperspectral_Analysis\cnn_model\models\CNN_3d25x25_e24_c8.pth',map_location='cuda:0')
    model.load_state_dict(state['model'])
    dataset = MoniHDF5_leaning_dataset(r'D:\Programing\pythonProject\data_store\eval_datasets_25_25.h5')
    dataLoader = DataLoader(dataset,pin_memory=True, shuffle=False, batch_size=12, num_workers=4)
    true_labels, all_preds = compute_confusion_matrix(model, dataLoader, label_nums=len(dataset))
    draw_confusion_matrix(true_labels, all_preds, class_name=None)

    # cm = np.load('Confusion_matrix.npz')['arr_0']
    # draw_confusion_matrix(cm=cm, class_name=None)