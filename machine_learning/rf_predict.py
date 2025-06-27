import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
from core import Hyperspectral_Image
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from utils import label_to_rgb
import matplotlib.pyplot as plt
if __name__ == '__main__':
    img = Hyperspectral_Image()
    img.init(r'D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat')
    img.image_enhance(f='PCA', n_components = 24)
    X = img.enhance_data[img.backward_mask]

    with open(r'D:\Programing\pythonProject\Hspectral_Analysis\machine_learning\rf_model.pkl', 'rb') as f:
        clf_loaded = pickle.load(f)
    # 用加载的模型进行预测
    y_pred = clf_loaded.predict(X)
    predict_map = np.zeros((img.rows, img.cols), dtype=np.int8) - 1
    predict_map[img.backward_mask] = y_pred
    color_map = label_to_rgb(predict_map)
    plt.imsave('rf_result.png', color_map, dpi=300)