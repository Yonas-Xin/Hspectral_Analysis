
import numpy as np
from core import Hyperspectral_Image, pca
import numpy as np
import os
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from gdal_utils import mask_to_vector_gdal, vector_to_mask
from sklearn.preprocessing import normalize
# from plotly_draw import show_scatter3d,create_scatter_3d,multiplot_scatter_3d
def load_deep_feature(path):
    dataset = np.load(path)
    return dataset['data']


def relabel_by_average_coordinates(mask):
    """
    根据标签的平均坐标重新排序标签
    """
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels != 0]
    avg_coords = {}
    for label in unique_labels:
        rows, cols = np.where(mask == label)
        avg_row = np.mean(rows)
        avg_col = np.mean(cols)
        avg_coords[label] = (avg_row, avg_col)
    sorted_labels = sorted(unique_labels,
                           key=lambda x: (avg_coords[x][0], avg_coords[x][1]))
    new_label_map = {old_label: new_label
                     for new_label, old_label in enumerate(sorted_labels, start=1)}
    relabeled_mask = np.zeros_like(mask)
    for old_label in unique_labels:
        relabeled_mask[mask == old_label] = new_label_map[old_label]

    return relabeled_mask

input_tif_file = r"C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat"
input_shp_file = 'ppi_result.shp'
deep_feature_npz_file = 'SSAR_re50_Embedding24.npz'
classes = 24
out_shp_file = f'C:\\Users\\85002\\OneDrive - cugb.edu.cn\\研究区地图数据\\样本点数据\\作图数据\\research1_gmm{classes}.shp'

if __name__ == '__main__':
    """数据读取，数据增强，样本随机选择"""
    img = Hyperspectral_Image()
    img.init(input_tif_file, init_fig=False) # 加载原始影像
    mask = img.create_mask(input_shp_file)
    print(np.sum(mask))
    dataset = load_deep_feature(deep_feature_npz_file) # 加载深度特征
    model = GaussianMixture(n_components=classes, n_init=10)
    labels = model.fit_predict(dataset) + 1 #不能让标签为0
    mask[mask != 0] = labels  # 样本类型更新
    mask = relabel_by_average_coordinates(mask) # 标签排序
    if os.path.exists(out_shp_file):
        raise ValueError(f'{out_shp_file} exists')
    else:img.create_vector(mask,out_shp_file)