import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
import pickle
from utils import label_to_rgb
import matplotlib.pyplot as plt

input_data_path = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
input_model_path = r'D:\Programing\pythonProject\Hspectral_Analysis\rf_model.pkl'
out_tif_path = r'rf_result.tif'
if __name__ == '__main__':
    print("Processing data...")
    img = Hyperspectral_Image()
    img.init(input_data_path)
    with open(input_model_path, 'rb') as f:
        clf_loaded = pickle.load(f)

    predict_whole_map = np.empty((img.rows,img.cols), dtype=np.int16) - 1 # 背景值为-1
    for image_block, background_mask, i, j in img.block_images(image_block=512, block_size=1):
        print(f'Processing block at position ({i}, {j}) with shape {image_block.shape}...')
        rows, cols = background_mask.shape
        predict_map = np.zeros((rows, cols), dtype=np.int16) - 1 # 初始化一个空的预测map，-1代表背景值
        if np.any(background_mask == True): # 如果
            idx = 0
            dataset = image_block.transpose(1,2,0)[background_mask] # 只对非背景像素进行预测
            predict_data = np.empty(len(dataset), dtype=np.int16) # 预分配内存，用来储存预测结果
            predict_map[background_mask] = clf_loaded.predict(dataset)
        predict_whole_map[i:i+rows, j:j+cols] = predict_map # 将预测结果填入整体预测矩阵
        print(f'Block at position ({i}, {j}) processed!')
    img.save_tif(out_tif_path, predict_whole_map, nodata=-1) # 保存为tif文件
    map = label_to_rgb(predict_whole_map)
    plt.imsave(out_tif_path[:-4]+'.png', map, dpi=300)
    print("The result has been saved:", out_tif_path)