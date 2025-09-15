"""将输入数据进行标准化并存储为tif文件，供机器学习使用"""
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
from sklearn.preprocessing import StandardScaler
import numpy as np

input_tif = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat'
out_tif = r'scalered.tif'

if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(input_tif)
    dataset = img.get_dataset().transpose(1,2,0)[img.backward_mask]
    scaler = StandardScaler()
    dataset = scaler.fit_transform(dataset)
    no_data = 0 if img.no_data is None else img.no_data
    outdata = np.full((img.rows, img.cols, img.bands), no_data, dtype=np.float32)
    outdata[img.backward_mask] = dataset
    img.save_tif(out_tif, outdata, nodata=no_data)