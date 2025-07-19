import sys, os
import warnings
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
from core import Hyperspectral_Image, pca
try:
    from osgeo import gdal,ogr,osr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
from tqdm import tqdm
from datetime import datetime
import time
import numpy as np
import os
import matplotlib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from gdal_utils import mask_to_vector_gdal, vector_to_mask
from sklearn.preprocessing import normalize
from utils import read_csv_to_matrix

def load_deep_feature(path):
    dataset = np.load(path)
    return dataset['data']
# 使标签顺序与类别特征一致
def standardize_labels(labels, centroids):
    """按聚类中心大小重新编号标签
    
    参数:
        labels: 原始标签数组，形状为 (n_samples,)，值应为[0,1,2,...]
        centroids: 聚类中心矩阵，形状为 (n_clusters, n_features)
    
    返回:
        standardized_labels: 重新编号后的标签数组，形状同labels
    """
    centroid_means = centroids.mean(axis=1)
    sorted_order = np.argsort(centroid_means)  # 例如：输入[2,0,1] → 输出[1,2,0]表示原标签1最小
    label_mapping = {old_label: new_label 
                    for new_label, old_label in enumerate(sorted_order)}
    return np.vectorize(label_mapping.get)(labels)

def get_valid_features(shp_path, raster_path):
    """
    获取有效要素（满足Emb_Idx>0或位于栅格范围内的要素）
    
    参数:
        shp_path: 点要素Shapefile路径
        raster_path: 参考栅格文件路径
    
    返回:
        tuple: (有效要素列表, 要素总数)
    """
    # 打开栅格文件获取空间范围
    raster = gdal.Open(raster_path)
    if raster is None:
        raise RuntimeError(f"无法打开栅格文件: {raster_path}")
    
    # 计算栅格的空间边界范围
    gt = raster.GetGeoTransform()
    x_min = gt[0]  # 最小X坐标
    x_max = gt[0] + gt[1] * raster.RasterXSize  # 最大X坐标
    y_min = gt[3] + gt[5] * raster.RasterYSize  # 最小Y坐标
    y_max = gt[3]  # 最大Y坐标
    raster = None  # 释放栅格资源
    
    # 打开Shapefile文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_path)
    if ds is None:
        raise RuntimeError(f"无法打开Shapefile文件: {shp_path}")
    
    layer = ds.GetLayer()
    layer_defn = layer.GetLayerDefn()  # 获取图层定义
    
    # 检查是否存在Emb_Idx字段
    emb_idx_field = layer_defn.GetFieldIndex('Emb_Idx')
    has_emb_idx = emb_idx_field != -1  # 标记是否存在该字段
    
    if not has_emb_idx:
        warnings.warn("警告：Shapefile中未找到Emb_Idx字段，将仅使用空间范围进行筛选")
    
    valid_features = []  # 存储有效要素
    for feature in layer:
        # 首先检查Emb_Idx字段是否有效
        if has_emb_idx:
            emb_idx = feature.GetField(emb_idx_field)
            # 如果Emb_Idx存在且大于0，视为有效要素
            if emb_idx is not None and emb_idx > 0:
                valid_features.append(feature.Clone())  # 克隆要素避免引用问题
                continue  # 跳过空间范围检查
        
        # 如果没有有效的Emb_Idx，则检查空间范围
        geom = feature.GetGeometryRef()
        if geom is None:
            continue  # 跳过无几何图形的要素
            
        # 获取要素坐标
        x, y = geom.GetX(), geom.GetY()
        # 检查是否在栅格范围内
        if x_min <= x <= x_max and y_min <= y <= y_max:
            valid_features.append(feature.Clone())
    
    total_count = layer.GetFeatureCount()  # 获取要素总数
    ds = None  # 释放数据源
    
    return valid_features, total_count



def split_shp_by_labels(input_shp, raster_path, labels, output_dir, output_dir_name=None):
    """
    根据标签拆分有效要素到多个Shapefile
    
    参数:
        input_shp: 输入点Shapefile路径
        raster_path: 参考栅格路径
        labels: 仅包含有效要素的标签列表
        output_dir: 输出目录
    
    返回:
        dict: {label: 输出文件路径}
    """
    # 获取有效要素
    valid_features, total_count = get_valid_features(input_shp, raster_path)
    
    # 验证标签数量匹配
    if len(labels) != len(valid_features):
        raise ValueError(
            f"标签数量({len(labels)})与有效要素数量({len(valid_features)})不匹配\n"
            f"原始要素总数: {total_count}"
        )
    
    # 准备输出文件
    os.makedirs(output_dir, exist_ok=True)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    
    # 获取原始字段定义和SRS
    sample_ds = driver.Open(input_shp)
    sample_layer = sample_ds.GetLayer()
    feature_defn = sample_layer.GetLayerDefn()
    srs = sample_layer.GetSpatialRef()
    sample_ds = None
    
    # 创建按标签分组的输出文件
    output_files = {}
    out_ds_dict = {}
    
    # 创建输出目录
    if output_dir_name is None:
        output_dir_name = 'SAMPLES_DIR'
    OUTPUT_DIR = os.path.join(output_dir, output_dir_name)
    counter = 1
    while os.path.exists(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(output_dir, f"{output_dir_name}_{counter}")
        counter += 1
    os.makedirs(OUTPUT_DIR)

    for label in tqdm(set(labels), desc="创建文件", total=len(set(labels))):
        time.sleep(1) # 确保间隔一秒创建一个文件
        current_time = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = os.path.join(OUTPUT_DIR, f"{current_time}_Point.shp")
        if os.path.exists(output_path):
            driver.DeleteDataSource(output_path)
        
        out_ds = driver.CreateDataSource(output_path)
        out_layer = out_ds.CreateLayer(f"points_{label}", srs=srs, geom_type=ogr.wkbPoint)
        
        # 复制字段
        for i in range(feature_defn.GetFieldCount()):
            out_layer.CreateField(feature_defn.GetFieldDefn(i))
        
        out_ds_dict[label] = (out_ds, out_layer)
        output_files[label] = output_path
    
    # 写入要素
    for feature, label in tqdm(zip(valid_features, labels), desc="写入要素", total=len(valid_features)):
        out_ds, out_layer = out_ds_dict[label]
        
        out_feature = ogr.Feature(out_layer.GetLayerDefn())
        out_feature.SetGeometry(feature.GetGeometryRef().Clone())
        
        for j in range(feature_defn.GetFieldCount()):
            out_feature.SetField(j, feature.GetField(j))
        
        out_layer.CreateFeature(out_feature)
        out_feature = None
    
    # 关闭所有文件
    for feature in valid_features:
        feature = None
        
    for ds, _ in out_ds_dict.values():
        ds = None
    print('Finish')
    return output_files

input_img = r'C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat'
input_shp = r'C:\Users\85002\Desktop\TempDIR\out.shp'
embeddings = r'D:\Data\Hgy\research_clip_samples\embeddings.csv'
classes = 24
out_dir = r'c:\Users\85002\Desktop\TempDIR\test2'
out_dir_name = 'RESEARCH_GF5'
if __name__ == '__main__':
    """数据读取，数据增强，样本随机选择"""
    dataset = read_csv_to_matrix(embeddings) 
    model = GaussianMixture(n_components=classes, n_init=10)

    labels = model.fit_predict(dataset)
    labels = standardize_labels(labels, model.means_) # 对标签进行标准化排序，确保每次聚类结果一致

    split_shp_by_labels(input_shp, input_img, labels,  output_dir=out_dir, output_dir_name=out_dir_name)