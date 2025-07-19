'''特征转换，将样本转化为低维编码
见contrastive_learning文件夹下Perdict_embedding文件'''
import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import torch
import numpy as np
from contrastive_learning.Models.Data import DynamicCropDataset
from torch.utils.data import DataLoader
from contrastive_learning.Models import Models
from utils import read_txt_to_list, save_matrix_to_csv
from tqdm import tqdm
from contrastive_learning.Models.Frame import Contrasive_learning_predict_frame
from core import Hyperspectral_Image
def set_embedding_idx(input_shp, input_tif):
    """为shp文件创建索引字段Emb_Idx,字段0为无编码点,字段1-n为有编码点"""
    im_dataset = gdal.Open(input_tif)
    im_geotrans = im_dataset.GetGeoTransform()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_dataset = driver.Open(input_shp, 1)  # 1 表示可写
    if shp_dataset is None:
        raise RuntimeError(f"无法打开矢量文件: {input_shp}")
        
    layer = shp_dataset.GetLayer()
    # 检查并删除现有Emb_Idx字段
    field_idx = layer.FindFieldIndex('Emb_Idx', 1)
    if field_idx != -1:
        layer.DeleteField(field_idx)
        print("已删除现有Emb_Idx字段")
    
    # 创建新的Emb_Idx字段
    embedding_field = ogr.FieldDefn('Emb_Idx', ogr.OFTInteger)
    if layer.CreateField(embedding_field) != 0:
        raise RuntimeError("创建Emb_Idx字段失败")
    # 获取新字段的索引
    layer_defn = layer.GetLayerDefn()
    field_idx = layer_defn.GetFieldIndex('Emb_Idx')
    if field_idx == -1:
        raise RuntimeError("无法找到新创建的 Emb_Idx 字段")
    idx = 1
    layer.ResetReading()  # 重置读取位置
    for feature in layer:
        geom = feature.GetGeometryRef()
        geoX, geoY = geom.GetX(), geom.GetY()
        
        # 转换为像素坐标
        x = int((geoX - im_geotrans[0]) / im_geotrans[1])
        y = int((geoY - im_geotrans[3]) / im_geotrans[5])
        
        # 检查是否在影像范围内
        if (0 <= x < im_width and 
            0 <= y < im_height):
            feature.SetField(field_idx, idx) # 设置点的编码索引
            idx += 1
        else: 
            feature.SetField(field_idx, 0)
        layer.SetFeature(feature)
    print('Has set Emb_Idx field')
    shp_dataset = None
    im_dataset = None

model_path = r'C:\Users\85002\Desktop\模型\模型pth与log\Spe_Spa_Attenres110_retrain_202504281258.pth'
input_img = r'C:\Users\85002\OneDrive - cugb.edu.cn\研究区地图数据\研究区影像数据\research_area1.dat'
input_shp = r'C:\Users\85002\Desktop\TempDIR\out.shp'

FUNC = "Deep_Learning" # Deep_Learning PCA MNF
embedding_dims = 24
if __name__ == '__main__':
    if FUNC == "Deep_Learning":
        device = torch.device('cuda')
        dataset = DynamicCropDataset(input_img, input_shp, block_size=17)
        dataloader = DataLoader(dataset, shuffle=False, batch_size=24)
        model = Models.Spe_Spa_Attenres(24, dataset.data_shape)  # 模型实例化
        state_dict = torch.load(model_path, weights_only=True, map_location=device)
        model.load_state_dict(state_dict['model'])
        frame = Contrasive_learning_predict_frame(device=device)
        out_embeddings = frame.predict(model, dataloader)
    elif FUNC == "PCA" or "MNF":
        set_embedding_idx(input_shp, input_img)
        img = Hyperspectral_Image()
        img.init(input_img, init_fig=True) # 需要加载掩膜
        img.image_enhance(f=FUNC, n_components=embedding_dims)
        mask = img.create_mask(input_shp)
        out_embeddings = img.enhance_data[mask>0]
    else:
        raise ValueError("the input must be: Deep_Learning PCA MNF")
    save_matrix_to_csv(out_embeddings, r'D:\Data\Hgy\research_clip_samples\embeddings2.csv')