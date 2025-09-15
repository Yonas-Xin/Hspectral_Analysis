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
import matplotlib
matplotlib.use('Agg')

class Block_Generator(Dataset):
    '''构造用于3D编码器的输入''' 
    def __init__(self, patch_size):
        self.block_size = patch_size
        self.data = None
        self.rows, self.cols = 0, 0
        self.idx = None

    def __len__(self):
        return len(self.idx) if self.idx is not None else 0
    
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        index = self.idx[idx]
        row = index // self.cols
        col = index % self.cols # 根据索引生成二维索引
        block = self.get_samples(row, col)
        # 转换为 PyTorch 张量
        block = torch.from_numpy(block).float()
        return block

    def get_samples(self,row,col):
        block = self.data[:,row:row + self.block_size, col:col + self.block_size]
        if self.block_size == 1: # 如果是单像素，数据适配1D CNN的输入
            block = block.squeeze()
        return block
    
    def update_data(self, data, background_mask=None):
        """每个 block 更新数据"""
        rows, cols = background_mask.shape
        self.data = data
        self.rows = rows
        self.cols = cols
        self.idx = np.arange(self.rows * self.cols)

        mask = background_mask.astype(np.bool)
        self.idx = self.idx.reshape(self.rows, self.cols)
        self.idx = self.idx[mask]

def create_img(img1, img2, outpath):
    """
    绘制原图和预测图对比图
    img1: [rows, cols, 3]
    img2: [rows, cols, 3]
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.imshow(img1)
    ax1.axis('off')
    ax2.imshow(img2)
    ax2.axis('off')
    fig.savefig(outpath, bbox_inches='tight')
    plt.close(fig)

def clean_up(output_dir):
    if len(os.listdir(output_dir)) == 0:
        print(f'the temp_dir {output_dir} has been deleted!')
        os.rmdir(output_dir)

output_path = 'SRACN_PRE.tif'
out_classes = 11
block_size = 17
batch = 128
input_data = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat"
model_pth = r'cnn_model\_results\SRACN_202509071837\SRACN_202509071837_best.pt'  # 模型路径
rgb_combine = (29,19,9) # 绘制图像时的rgb组合，从1开始
if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image_block_size = 512
    left_top = int(block_size / 2 - 1) if block_size % 2 == 0 else int(block_size // 2)
    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
    output_dir = '.\\cnn_model\\temp_dir' # 临时文件夹，保存中间图片
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img = Hyperspectral_Image()
    img.init(input_data, rgb=rgb_combine)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16) - 1 # 背景值为-1
    model = torch.load(model_pth, weights_only=False, map_location=device)
    model.to(device)
    model.eval()

    dataset = Block_Generator(block_size)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=0) # 实测这里numworker变多不会加快速率，反而会因为加载进程拖慢速度
    try:
        with torch.no_grad():
            for image_block, background_mask, i, j in img.block_images(image_block=image_block_size, block_size=block_size):
                rows, cols = background_mask.shape
                predict_map = np.zeros((rows, cols), dtype=np.int16) - 1 # 初始化一个空的预测map，-1代表背景值
                if np.any(background_mask == True): # 如果
                    idx = 0
                    dataset.update_data(image_block, background_mask) # 更新dataset后调用dataloader会重新启动进程
                    predict_data = torch.empty(len(dataset), dtype=torch.int16, device=device) # 预分配内存，用来储存预测结果
                    for data in tqdm(dataloader, total=len(dataloader), desc=f'Block{i}_{j}'):
                        batch = data.shape[0]
                        data = data.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs, 1)
                        predict_data[idx:idx + batch, ] = predicted
                        idx += batch
                predict_map[background_mask] = predict_data.cpu().numpy() if predict_data.device.type == 'cuda' else predict_data.numpy() # 将预测结果填入对应位置
                predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵
                img.save_tif(output_path, predict_whole_map, nodata=-1) # 保存为tif文件

                # 下面保存预测过程中的图像
                map = utils.label_to_rgb(predict_whole_map)
                out_png = os.path.join(output_dir, f"{output_path[:-4]}-{current_time}.png")
                create_img(img.ori_img, map, out_png)
    except KeyboardInterrupt as k:
        clean_up(output_dir)
    except Exception as e:
        clean_up(output_dir)
        print(traceback.format_exc()) 