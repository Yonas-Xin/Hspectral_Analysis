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
from cnn_model.Models.Models import MODEL_DICT
import matplotlib
from datetime import datetime
import signal
from multiprocessing import cpu_count
matplotlib.use('Agg')  # 使用非GUI后端，彻底绕过tkinter

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
            if backward_mask.dtype != np.bool:
                backward_mask = backward_mask.astype(np.bool)
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
        if self.block_size == 1: # 如果是单像素，数据适配1D CNN的输入
            block = block.squeeze()
        return block

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
    
if __name__ == '__main__':
    model_name = "Shallow_1DCNN"
    out_classes = 15
    block_size = 1
    batch = 256
    input_data = r"D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_pth = 'D:\Programing\pythonProject\Hspectral_Analysis\cnn_model\_results\models_pth\SSAR_202506261846.pth'  # 模型路径
    csv_output_path = 'out.csv'


    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    data_shape = None
    model = None
    image_block_size = 512
    out_embedding = 24
    def interrupt_handler(signum, frame):
        print("\nInterrupt signal received.")
        clean_up()
        exit(1)
    def clean_up():
        if len(os.listdir(output_dir)) == 0:
            print(f'the temp_dir {output_dir} has been deleted!')
            os.rmdir(output_dir)
    signal.signal(signal.SIGINT, interrupt_handler)  # 注册中断信号处理函数
    signal.signal(signal.SIGTERM, interrupt_handler)  # 注册终止信号处理函数

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
    img = Hyperspectral_Image()
    img.init(input_data)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16)
    try:
        with torch.no_grad():
            for image_block, background_mask, i, j in img.block_images(image_block=image_block_size, block_size=block_size):
                dataset = Block_Generator(image_block, block_size=block_size, backward_mask=background_mask)
                predict_data = torch.empty(len(dataset), dtype=torch.int16, device=device) # 预分配内存，用来储存预测结果
                rows, cols = dataset.rows, dataset.cols

                if data_shape is None:
                    data_shape = dataset.image_shape

                if model is None: # 进行模型的初始化和参数读取
                    model = MODEL_DICT['model_name'](out_embedding=out_embedding, out_classes=out_classes, in_shape=data_shape) # 在这里进行模型初始化
                    if model_pth is not None:
                        dic = torch.load(model_pth, weights_only=True, map_location=device)['model']
                        model.load_state_dict(dic)
                model.to(device)
                model.eval()

                predict_map = np.zeros((rows, cols), dtype=np.int16) - 1 # 初始化一个空的预测矩阵，-1代表背景值
                if np.any(background_mask == True):
                    idx = 0
                    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True,num_workers=dataloader_num_workers,prefetch_factor=2)
                    for data in tqdm(dataloader, total=len(dataloader), desc=f'Block{i}_{j}'):
                        batch = data.shape[0]
                        data = data.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs, 1)
                        predict_data[idx:idx + batch, ] = predicted
                        idx += batch
                predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
                predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵

                # 下面保存预测过程中的图像
                map = utils.label_to_rgb(predict_map)
                image = image_block[(29,19,9),left_top:left_top+rows,left_top:left_top+cols].transpose(1,2,0)
                output_path = os.path.join(output_dir, f"block_{i}_{j}.png")
                save_img(image, map, output_path)
    except Exception as e:
        clean_up()
        raise (e)
    utils.save_matrix_to_csv(predict_whole_map, csv_output_path) # 保存为csv文件