from core import Hyperspectral_Image,pca
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
# from base_utils.plotly_draw import show_scatter3d,create_scatter_3d,multiplot_scatter_3d
'''颜色条'''
VOC_COLORMAP = [[255, 255, 255], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]
def gmm_cluster(data, mask, n_components=12):
    '''gmm聚类,返回具有标签的mask'''
    model = GaussianMixture(n_components=n_components)
    labels = model.fit_predict(data) + 1  # 不能让标签为0
    mask[mask > 0] = labels  # 样本类型更新
    return mask

def kmeans_cluster(data, mask, n_components=12):
    model = KMeans(n_clusters=n_components)
    labels = model.fit_predict(data) + 1  # 不能让标签为0
    mask[mask > 0] = labels  # 样本类型更新
    return mask

def load_deep_feature(path):
    dataset = np.load(path)
    return dataset['data']

input_tif_file = r"C:\Users\85002\Desktop\毕设\research_area1.dat"
input_shp_file = r'C:\Users\85002\Desktop\模型\research1_SSARre50_deep24_gmm24.shp'
deep_feature_npz_file = 'D:\Programing\pythonProject\Hyperspectral_Analysis\SSAR_re50_Embedding24.npz'
out_shp_file = f'C:\\Users\\85002\\Desktop\\模型\\research1_SSARre50_deep24_gmm24_optimization2.shp'

nums = 3 # 过滤次数为3
if __name__ == '__main__':
    """数据读取，数据增强，样本随机选择"""
    img = Hyperspectral_Image()
    img.init(input_tif_file) # 加载原始影像
    mask = img.create_mask(input_shp_file) # 加载样本点位，如果将所有像素视作样本点则调用img.backward_mask()
    dataset = load_deep_feature(deep_feature_npz_file)
    idx = mask[mask > 0]    # 如果是8225数据改成 idx = mask[mask ！= 0]
    print(f"原始样本数量：{len(idx)}")

    labels = np.unique(mask)
    for j in range(nums):
        for i in labels:
            if i == 0 :
                continue
            label_mask = (idx==i)
            data = dataset[label_mask]
            if data.shape[0]<100:
                continue
            label = np.zeros(data.shape[0])+i
            model = IsolationForest(n_estimators=1000,
                                    max_samples='auto',
                                    contamination=float(0.1),
                                    max_features=1.0)
            x = model.fit_predict(data)
            label[x==-1] = -1
            idx[idx==i] = label # 孤立点位置标记为-1
        print(f"浓缩样本数量：{len(idx[idx > 0])}")
        mask[mask != 0] = idx # 更新mask
    img.create_vector(mask, out_shp_file)