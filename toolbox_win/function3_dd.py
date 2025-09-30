import sys, os
sys.path.append('.')
try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import torch
from contrastive_learning.Models.Data import DynamicCropDataset
from torch.utils.data import DataLoader
from utils import save_matrix_to_csv
from contrastive_learning.Models.Frame import Contrasive_learning_predict_frame
from core import Hyperspectral_Image
import argparse

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

def dd_prediction(input_tif, input_shp, model_path, output_csv, FUNC="DL", patch_size=17, batch_size=256, embedding_dims=24):
    """使用对比学习模型进行降维并保存结果
    
    参数:
    - input_tif: 输入的高光谱影像路径
    - input_shp: 输入的点矢量路径，点矢量中必须包含Emb_Idx字段
    - model_path: 训练好的模型路径
    - output_csv: 输出的csv文件路径
    - FUNC: 降维方法，"DL"表示对比学习，"PCA"表示主成分分析
    - patch_size: 样本裁剪大小
    - batch_size: 批处理大小
    - embedding_dims: 降维后的维度
    """
    try:
        if FUNC == "DL":
            device = torch.device('cuda')
            dataset = DynamicCropDataset(input_tif, input_shp, patch_size=patch_size)
            dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
            model = torch.load(model_path, weights_only=False)
            frame = Contrasive_learning_predict_frame(device=device)
            out_embeddings = frame.predict(model, dataloader)
        elif FUNC == "PCA" or "MNF":
            set_embedding_idx(input_shp, input_tif)
            img = Hyperspectral_Image()
            img.init(input_tif, init_fig=True) # 需要加载掩膜
            img.image_enhance(f=FUNC, n_components=embedding_dims)
            mask = img.create_mask(input_shp)
            out_embeddings = img.enhance_data[mask>0]
        save_matrix_to_csv(out_embeddings, output_csv)
        return True, "降维成功"
    except Exception as e:
        return False, f"降维失败: {e}"

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--input_tif", type=str, required=True)
    argparser.add_argument("--input_shp", type=str, required=True)
    argparser.add_argument("--output_csv", type=str, required=True)
    argparser.add_argument("--FUNC", type=str, default="DL")
    argparser.add_argument("--model_path", type=str)
    argparser.add_argument("--patch_size", type=int, default=17)
    argparser.add_argument("--batch_size", type=int, default=256)
    argparser.add_argument("--embedding_dims", type=int, default=24)
    args = argparser.parse_args()

    dd_prediction(
        input_tif=os.path.abspath(args.input_tif),
        input_shp=os.path.abspath(args.input_shp),
        model_path=None if args.model_path == "None" else os.path.abspath(args.model_path),
        output_csv=os.path.abspath(args.output_csv),
        FUNC=args.FUNC,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        embedding_dims=args.embedding_dims
    )