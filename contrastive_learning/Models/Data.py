import numpy as np
import torch
from torch.utils.data import Dataset
from threading import Lock
try:
    import h5pickle
except:
    pass

try:
    from osgeo import gdal, ogr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')


def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据
    返回dataset[bands,H,W]'''
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) * 1e-4
    return dataset

class SSF(Dataset):
    '''输入两个个list文件，list元素代表数据地址，一个存放窗口，一个存放光谱'''
    def __init__(self, data_list, spectral_list, transform=None):
        """
        将列表划分为数据集,[batch, C, H, W] [batch, 1, sequence]
        """
        self.image_paths = data_list
        self.spectral_lists = spectral_list

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（3，rows，cols）
        """
        image_path = self.image_paths[idx]
        spectral_path = self.spectral_lists[idx]
        image = read_tif_with_gdal(image_path)
        spectral = read_tif_with_gdal(spectral_path).transpose(1,2,0)
        spectral = np.squeeze(spectral, axis=0)

        # 转换为 PyTorch 张量
        image = torch.from_numpy(image).float()
        spectral = torch.from_numpy(spectral).float()
        return image, spectral

class Dataset_3D(Dataset):
    '''输入一个list文件，list元素代表数据地址'''
    def __init__(self, data_list, transform=None):
        """
        将列表划分为数据集,[batch, 1, H, w, bands]
        """
        self.image_paths = data_list
        image = self.__getitem__(0)
        self.data_shape = image.shape

    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, idx):
        """
        根据索引返回图像及其标签
        image（3，rows，cols）
        """
        image_path = self.image_paths[idx]
        image = read_tif_with_gdal(image_path)
        image = torch.from_numpy(image).float()
        return image

class Dataset_3D_H5(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        self.hf = h5pickle.File(self.h5_file, 'r')  # 只读模式
        self.data = self.hf['data']
    def __len__(self):
        """返回数据集大小"""
        return len(self.data)
    def __getitem__(self, index):
        """
        获取单个样本
        :param index: 样本索引
        :return: (影像数据, 标签)
        """
        img = self.data[index]  # 读取 HDF5 数据
        # img = torch.tensor(img, dtype=torch.float32)  # 转换为 Tensor
        return img
    def close(self):
        """关闭 HDF5 文件"""
        self.hf.close()


class DynamicCropDataset(Dataset):
    """
    动态从原始影像和点Shapefile生成训练数据的Dataset
    
    参数:
        image_path (str): 原始影像路径
        point_shp (str): 点Shapefile路径
        block_size (int): 裁剪块大小（像素）
        transform (callable): 可选的数据增强变换
        fill_value (int/float): 边缘填充值
    """
    
    def __init__(self, image_path, point_shp, block_size=30, 
                 transform=None, fill_value=0):
        self.image_path = image_path
        self.point_shp = point_shp
        self.block_size = block_size
        self.transform = transform
        self.fill_value = fill_value
        
        # 初始化GDAL资源
        self.im_dataset = gdal.Open(image_path)
        if self.im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {image_path}")
            
        # 获取影像基本信息
        self.im_geotrans = self.im_dataset.GetGeoTransform()
        self.im_proj = self.im_dataset.GetProjection()
        self.im_width = self.im_dataset.RasterXSize
        self.im_height = self.im_dataset.RasterYSize
        self.im_bands = self.im_dataset.RasterCount
        
        # 加载所有有效点坐标
        self.point_coords = self._load_point_coordinates()

        # 获取数据形状
        image = self.__getitem__(0)
        self.data_shape = image.shape 

    
    def _load_point_coordinates(self):
        """依次加载所有有效的点坐标（影像范围内的点）,为shp文件创建索引字段Emb_Idx,字段0为无编码点,字段1-n为有编码点"""
        coords = []
        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = driver.Open(self.point_shp, 1)  # 1 表示可写
        if shp_dataset is None:
            raise RuntimeError(f"无法打开矢量文件: {self.point_shp}")
            
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
            x = int((geoX - self.im_geotrans[0]) / self.im_geotrans[1])
            y = int((geoY - self.im_geotrans[3]) / self.im_geotrans[5])
            
            # 检查是否在影像范围内
            if (0 <= x < self.im_width and 
                0 <= y < self.im_height):
                coords.append((x, y))
                feature.SetField(field_idx, idx) # 设置点的编码索引
                idx += 1
            else: 
                feature.SetField(field_idx, 0)
            layer.SetFeature(feature)
        print('Has set Emb_Idx field')
        shp_dataset = None
        if not coords:
            raise RuntimeError("没有找到影像范围内的有效点")
        return coords
    
    def __len__(self):
        return len(self.point_coords)
    
    def __getitem__(self, idx):
        """动态裁剪并返回数据块"""
        x, y = self.point_coords[idx]
        
        # 计算裁剪窗口
        if self.block_size % 2 == 0:
            left_top = self.block_size // 2 - 1
            right_bottom = self.block_size // 2
        else:
            left_top = right_bottom = self.block_size // 2
            
        x_start = x - left_top
        y_start = y - left_top
        x_end = x + right_bottom + 1
        y_end = y + right_bottom + 1
        
        # 计算实际可读取范围
        read_x = max(0, x_start)
        read_y = max(0, y_start)
        read_width = min(x_end, self.im_width) - read_x
        read_height = min(y_end, self.im_height) - read_y
        
        # 创建填充数组
        if self.im_bands > 1:
            block = np.full((self.im_bands, self.block_size, self.block_size), 
                           self.fill_value, dtype=np.float32)
        else:
            block = np.full((self.block_size, self.block_size), 
                          self.fill_value, dtype=np.float32)
        
        # 读取并填充有效数据
        if read_width > 0 and read_height > 0:
            if self.im_bands > 1:
                data = self.im_dataset.ReadAsArray(read_x, read_y, read_width, read_height)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) * 1e-4
                offset_x = read_x - x_start
                offset_y = read_y - y_start
                for b in range(self.im_bands):
                    block[b, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data[b]
            else:
                data = self.im_dataset.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) * 1e-4
                offset_x = read_x - x_start
                offset_y = read_y - y_start
                block[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        
        # 转换为torch张量
        block = torch.from_numpy(block)
        
        # 应用数据增强
        if self.transform:
            block = self.transform(block)
            
        return block
    
    def __del__(self):
        """释放GDAL资源"""
        if hasattr(self, 'im_dataset') and self.im_dataset:
            self.im_dataset = None