"""将大幅高光谱影像进行分块的滑窗预测，避免占用大量显存
预测结果是一个二维矩阵，-1代表背景, 其余值代表预测的地物类别"""
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
from datetime import datetime
import traceback
from multiprocessing import cpu_count
import matplotlib
matplotlib.use('Agg')

class Block_Generator(Dataset):
    '''构造用于3D编码器的输入'''
    def __init__(self, data, block_size, backward_mask=None):
        """分块滑窗Dataset"""
        _, act_rows, act_cols = data.shape
        self.data = data
        self.block_size = block_size
        left_top = int(block_size / 2 - 1) if block_size % 2 == 0 else int(block_size // 2)
        right_bottom = int(block_size / 2) if block_size % 2 == 0 else int(block_size // 2)
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

def create_img(img1, img2, outpath):
    """绘制原图和预测图对比图"""
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(img2)
    plt.axis('off')
    plt.savefig(outpath, bbox_inches='tight', dpi=300)
    plt.close()

def clean_up(output_dir):
    if len(os.listdir(output_dir)) == 0:
        print(f'the temp_dir {output_dir} has been deleted!')
        os.rmdir(output_dir)

if __name__ == '__main__':
    model_name = "SRACN"
    out_classes = 8
    block_size = 17
    batch = 256
    input_data = r"c:\Projects\Hspectral_Analysis\ZY_dataset\research_area1.dat"
    model_pth = r'C:\123pan\Downloads\SRACN-spectral0.5_band0_noise0.5_s50_Emd128_202508271613_best.pt'  # 模型路径
    csv_output_path = 'SRACN-spectral0.5_band0_noise0.5_s50_Emd128.tif'
    rgb_combine = (29,19,9) # 绘制图像时的rgb组合Z


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataloader_num_workers = cpu_count() // 4 # 根据cpu核心数自动决定num_workers数量
    image_block_size = 512
    left_top = int(block_size / 2 - 1) if block_size % 2 == 0 else int(block_size // 2)
    right_bottom = int(block_size / 2) if block_size % 2 == 0 else int(block_size // 2)
    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
    output_dir = f'.\\cnn_model\\temp_dir\\{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Hyperspectral_Image()
    img.init(input_data)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16)
    model = torch.load(model_pth, weights_only=False, map_location=device)
    try:
        with torch.no_grad():
            for image_block, background_mask, i, j in img.block_images(image_block=image_block_size, block_size=block_size):
                dataset = Block_Generator(image_block, block_size=block_size, backward_mask=background_mask)
                predict_data = torch.empty(len(dataset), dtype=torch.int16, device=device) # 预分配内存，用来储存预测结果
                rows, cols = dataset.rows, dataset.cols

                predict_map = np.zeros((rows, cols), dtype=np.int16) - 1 # 初始化一个空的预测矩阵，-1代表背景值
                if np.any(background_mask == True): # 如果
                    model.to(device)
                    model.eval()
                    idx = 0
                    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=dataloader_num_workers)
                    for data in tqdm(dataloader, total=len(dataloader), desc=f'Block{i}_{j}'):
                        batch = data.shape[0]
                        data = data.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs, 1)
                        predict_data[idx:idx + batch, ] = predicted
                        idx += batch
                predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
                predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵
                img.save_tif(csv_output_path, predict_whole_map, nodata=-1) # 保存为tif文件

                # 下面保存预测过程中的图像
                map = utils.label_to_rgb(predict_map)
                image = image_block[rgb_combine,left_top:left_top+rows,left_top:left_top+cols].transpose(1,2,0)
                output_path = os.path.join(output_dir, f"block{i}-{i+rows}_{j}-{j+cols}.png")
                create_img(image, map, output_path)
    except KeyboardInterrupt as k:
        clean_up(output_dir)
    except Exception as e:
        clean_up(output_dir)
        print(traceback.format_exc()) 