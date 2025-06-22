import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import utils
from core import Hyperspectral_Image
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from cnn_model.Models.Models import Constrastive_learning_Model
from datetime import datetime
class IS_Generator(Dataset):
    '''构造用于1d+2d编码器的输入'''
    def __init__(self, whole_space, whole_spectral, block_size=25):
        '''将整幅图像裁剪成为适用于模型输入的数据集形式
        whole_space[H,W,3]'''
        _, self.rows, self.cols = whole_space.shape
        self.block_size = block_size
        if block_size % 2 == 0:
            left_top = int(block_size / 2 - 1)
            right_bottom = int(block_size / 2)
        else:
            left_top = int(block_size // 2)
            right_bottom = int(block_size // 2)
        self.whole_space = np.pad(whole_space, [(left_top, right_bottom), (left_top, right_bottom), (0, 0)], 'constant')
        self.whole_spectral = whole_spectral

    def __len__(self):
        return self.rows*self.cols
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        row = idx//self.cols
        col = idx%self.cols # 根据索引生成二维索引
        block, spectral = self.get_samples(row, col)

        # 转换为 PyTorch 张量
        block = torch.from_numpy(block).float()
        spectral = torch.from_numpy(spectral).float()
        return block, spectral

    def get_samples(self,row,col):
        block = self.whole_space[:, row:row + self.block_size, col:col + self.block_size]
        spectral = self.whole_spectral[:, row, col:col+1]
        return block,spectral

class Block_Generator(Dataset):
    '''构造用于3D编码器的输入'''
    def __init__(self, data, block_size, backward_mask=None):
        '''将整幅图像裁剪成为适用于模型输入的数据集形式
        data[C,H,W]'''
        _, act_rows, act_cols = data.shape
        self.data = data
        self.block_size = block_size
        if block_size % 2 == 0:
            left_top = int(block_size / 2 - 1)
            right_bottom = int(block_size / 2)
        else:
            left_top = int(block_size // 2)
            right_bottom = int(block_size // 2)
        self.left_top = left_top
        self.right_bottom = right_bottom
        rows = act_rows - left_top - right_bottom
        cols = act_cols - left_top - right_bottom
        self.rows, self.cols = rows, cols
        self.idx = np.arange(rows * cols)
        image = self.__getitem__(0)
        self.image_shape = image.shape
        if backward_mask is not None:
            if self.backward_mask.dtype != np.bool:
                self.backward_mask = self.backward_mask.astype(np.bool)
            self.idx = self.idx.reshape(rows, cols)
            self.idx = self.idx[background_mask]

    def __len__(self):
        return len(self.idx)
    
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        index = self.idx[idx]
        row = index//self.cols
        col = index%self.cols # 根据索引生成二维索引
        block = self.get_samples(row, col)
        # 转换为 PyTorch 张量
        block = torch.from_numpy(block).float()
        return block

    def get_samples(self,row,col):
        block = self.data[:,row:row + self.block_size, col:col + self.block_size]
        return block
    
def show_img(data:torch.Tensor):
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    image = data[(29,19,9),:,:].transpose(1,2,0)
    plt.imshow(image)
    plt.show()

def save_img(img1, img2, outpath):
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis('off')
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close()

def init_SSAR_model(out_embedding=24, out_classes=16, in_shape=None, model_pth=None, device=None):
    model = Constrastive_learning_Model(out_embedding=out_embedding, out_classes=out_classes, in_shape=in_shape)
    if model_pth is not None:
        dic = torch.load(model_path, weights_only=True, map_location=device)['model']
        model.load_state_dict(dic)
    model.to(device)
    return model

if __name__ == '__main__':
    input_data = r"D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_block_size=512
    block_size=17
    batch = 256
    model_pth = '.models/contrastive_learning_model.pth'  # 模型路径
    out_embedding=24
    out_classes=16
    data_shape = None


    if block_size % 2 == 0:
        left_top = int(block_size / 2 - 1)
        right_bottom = int(block_size / 2)
    else:
        left_top = int(block_size // 2)
        right_bottom = int(block_size // 2)
    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
    output_dir = f'.\\cnn_model\\temp_dir\\{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path  = '.models/'
    img = Hyperspectral_Image()
    img.init(input_data)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16)
    idx = 0
    with torch.no_grad():
        for image_block, background_mask, i, j in img.block_images(image_block=image_block_size, block_size=block_size):
            show_img(image_block)  # 展示滑动窗口
            dataset = Block_Generator(image_block, block_size=block_size)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True,num_workers=4,prefetch_factor=2)
            predict_data = torch.empty(len(dataset), dtype=torch.int16, device=device) # 预分配内存，用来储存预测结果
            rows, cols = dataset.rows, dataset.cols

            if data_shape is None:
                data_shape = dataset.image_shape
            model = Constrastive_learning_Model(out_embedding=out_embedding, out_classes=out_classes, in_shape=data_shape) # 在这里进行模型初始化
            if model_pth is not None:
                dic = torch.load(model_path, weights_only=True, map_location=device)['model']
                model.load_state_dict(dic)
            model.to(device)
            model.eval()

            for data in tqdm(dataloader, total=len(dataloader), desc=f'Block{i}_{j}'):
                batch = data.shape[0]
                # show_img(data[0,:,:,:] .cpu())# 展示滑动窗口
                data = data.unsqueeze(1).to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                predict_map[idx:idx + batch, ] = predicted
                idx += batch
            predict_map = np.zeros((rows, cols), dtype=np.int16) - 1 # 初始化一个空的预测矩阵，-1代表背景值
            predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
            predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵

            # 下面保存预测过程中的图像
            map = utils.label_to_rgb(predict_map)
            image = image_block[(29,19,9),left_top:left_top+rows,left_top:left_top+cols].transpose(1,2,0)
            output_path = os.path.join(output_dir, f"block_{i}_{j}.png")
            save_img(image, map, output_path)
    utils.save_matrix_to_csv(predict_whole_map, 'predict_map25_25.npz') # 保存为csv文件