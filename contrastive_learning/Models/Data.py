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
import random


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
                block[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
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



class Im_info(object):
    """用于计算影像掩膜，记录影像信息"""
    def __init__(self, image_path):
        self.image_path = image_path
        self.dataset = gdal.Open(image_path, gdal.GA_ReadOnly)
        band = self.dataset.GetRasterBand(1)
        self.no_data = band.GetNoDataValue()
        self.rows, self.cols = self.dataset.RasterYSize, self.dataset.RasterXSize
        self.backward_mask = self.ignore_backward()
        self.im_width = self.dataset.RasterXSize # 影像宽，W
        self.im_height = self.dataset.RasterYSize # 影像高，H
        self.im_bands = self.dataset.RasterCount # 波段数

    def ignore_backward(self):
        '''分块计算背景掩膜值，默认分块大小为512'''
        block_size = 512
        if self.cols> (2 * block_size) and self.rows > (2 * block_size):
            pass
        else:
            block_size = min(self.rows, self.cols) # 如果行列都较小，则使用行列最小值作为分块大小
        mask = np.empty((self.rows, self.cols), dtype=bool)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(block_size, self.rows - i)
                actual_cols = min(block_size, self.cols - j)
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                block_mask = np.all(block_data == self.no_data, axis=0)
                mask[i:i + actual_rows, j:j + actual_cols] = ~block_mask
        return mask
    
    def __del__(self):
        """释放GDAL资源"""
        if hasattr(self, 'dataset') and self.dataset:
            self.dataset = None

class MultiImageRandomBlockSampler(object):
    """多图像分块随机裁剪采样器，用于从多个图像中分区域随机选择图像块"""
    def __init__(self, image_paths_list, block_size):
        self.image_paths_list = image_paths_list
        self.im_info_objs = [Im_info(image_path) for image_path in image_paths_list]
        self.block_size = block_size
        self.idx_list = [] # [(Im_info: np.array([0,2,3,...])),...]
        self.out_list = [] # [(Im_info: 1), (Im_info: 98)...]
        self.iters = 0

        self.cal_area()
        self.reset()
    
    def cal_area(self):
        """计算每个数据的分块区域"""
        image_block = self.block_size
        for obj in self.im_info_objs:
            mask = obj.backward_mask
            nums = mask.size
            rows, cols = mask.shape
            idx_matrix = np.arange(nums).reshape(mask.shape)
            for i in range(0, rows, image_block):
                for j in range(0, cols, image_block):
                    # 计算当前块的实际高度和宽度（避免越界）
                    actual_rows = min(image_block, rows - i)  # 实际高
                    actual_cols = min(image_block, cols - j)  # 实际宽
                    block_idx = idx_matrix[i:i + actual_rows, j:j+actual_cols].reshape(-1)
                    mask_idx = mask[i:i + actual_rows, j:j+actual_cols].reshape(-1)
                    if np.any(mask_idx):
                        block_idx = block_idx[mask_idx]
                        self.idx_list.append((obj, block_idx))
                    else:
                        continue
        self.iters = len(self.idx_list)
    
    def reset(self):
        self.out_list = [(i[0], random.choice(i[1])) for i in self.idx_list]

class MIRBS_Dataset(Dataset):
    """多图像动态分块随机裁剪数据集，使用该类时请设置numworkers=0（经过实验，预先读取gdal数据会使得
    加载数据更快，比多进程更快）。不适用多进程，应该也不适用分布式训练，如有需要，做一些适当的修改即可"""
    def __init__(self, 
                 image_paths_list, # 图像路径列表
                 block_size):
        self.MIRBS = MultiImageRandomBlockSampler(image_paths_list, block_size)
        self.block_size= block_size
        self.list = self.MIRBS.out_list
        image = self.__getitem__(0)
        self.data_shape = image.shape
    
    def reset(self): # 重置索引,外部调用
        self.MIRBS.reset()
        self.list = self.MIRBS.out_list

    def __len__(self):
        return self.MIRBS.iters
    
    def __getitem__(self, idx):
        obj, index = self.list[idx]
        im_width = obj.im_width
        im_height = obj.im_height
        im_bands = obj.im_bands
        dataset = obj.dataset # 预先读取的gdal数据集，如果要实现多进程，这里修改为dataset = gdal.Open(obj.image_path)并删掉obj的dataset属性
        row = index // im_width # 计算行索引
        col = index % im_width # 计算列索引

        # 计算裁剪窗口
        block_size = self.block_size
        left_top = int(block_size / 2 - 1) if block_size % 2 == 0 else int(block_size // 2)
        right_bottom = int(block_size / 2) if block_size % 2 == 0 else int(block_size // 2)
        col_start = col - left_top
        row_start = row - left_top
        col_end = col + right_bottom + 1
        row_end = row + right_bottom + 1
        
        # 计算实际可读取范围
        read_col = max(0, col_start)
        read_row = max(0, row_start)
        read_width = min(col_end, im_width) - read_col
        read_height = min(row_end, im_height) - read_row
        
        # 创建并填充数组
        block = np.full((im_bands, self.block_size, self.block_size), 
                           0, dtype=np.float32)
        if read_width > 0 and read_height > 0:
            data = dataset.ReadAsArray(read_col, read_row, read_width, read_height)
            if data.dtype == np.int16:
                data = data.astype(np.float32) * 1e-4
            offset_x = read_col - col_start
            offset_y = read_row - row_start
            block[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        block = torch.from_numpy(block)
        return block