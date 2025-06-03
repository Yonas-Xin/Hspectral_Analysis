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
    def __init__(self, data, block_size=25):
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
        self.rows = act_rows - left_top - right_bottom
        self.cols = act_cols - left_top - right_bottom


    def __len__(self):
        return self.rows * self.cols
    def __getitem__(self, idx):
        """
        根据索引返回图像及其光谱
        """
        row = idx//self.cols
        col = idx%self.cols # 根据索引生成二维索引
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

if __name__ == '__main__':
    from cnn_model.Models.Models import Constrastive_learning_Model
    from datetime import datetime

    image_block_size=512
    block_size=17

    batch = 256
    if block_size % 2 == 0:
        left_top = int(block_size / 2 - 1)
        right_bottom = int(block_size / 2)
    else:
        left_top = int(block_size // 2)
        right_bottom = int(block_size // 2)
    current_time = datetime.now().strftime("%Y%m%d%H%M")  # 记录系统时间
    output_dir = f'.\\temp_dir\\{current_time}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path  = '.models/'
    img = Hyperspectral_Image()
    img.init(r"C:\Users\85002\Desktop\毕设\research_area1.dat", init_fig=False)

    model = Constrastive_learning_Model(out_embedding=24, out_classes=8, in_shape=(138,17,17))
    # dic = torch.load(model_path, weights_only=True)['model']
    # model.load_state_dict(dic)
    device = torch.device('cuda')
    model.to(device)
    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16)
    idx = 0
    model.eval()
    with torch.no_grad():
        for image_block,i,j in img.block_images(image_block=image_block_size, block_size=block_size, scale=1e-4):
            _, act_rows, act_cols = image_block.shape
            rows = act_rows-block_size+1
            cols = act_cols-block_size+1
            predict_map = torch.empty(rows*cols, dtype=torch.int16, device=device)
            # show_img(image_block)  # 展示滑动窗口
            dataset = Block_Generator(image_block, block_size=block_size)
            dataloader = DataLoader(dataset, batch_size=batch, shuffle=False, pin_memory=True,num_workers=4,prefetch_factor=2)

            for data in tqdm(dataloader, total=len(dataloader), desc=f'Block{i}_{j}'):
                batch = data.shape[0]
                # show_img(data[0,:,:,:] .cpu())# 展示滑动窗口
                data = data.unsqueeze(1).to(device)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                predict_map[idx:idx + batch, ] = predicted
                idx += batch
            predict_map = predict_map.reshape(rows, cols).cpu().numpy()
            predict_whole_map[i:i+rows, j:j+cols] = predict_map

            map = utils.label_to_rgb(predict_map)
            image = image_block[(29,19,9),left_top:left_top+rows,left_top:left_top+cols].transpose(1,2,0)
            output_path = os.path.join(output_dir, f"block_{i}_{j}.png")
            save_img(image, map, output_path)
    np.savez_compressed('predict_map25_25.npz',data=predict_whole_map)