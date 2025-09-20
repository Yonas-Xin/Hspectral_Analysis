"""
Image processing and display utilities for raster data
解决GDAL图像读取、转换和显示相关问题的工具模块
"""

import os
import sys
import warnings
from typing import Optional, Tuple, Union, Any
import codecs

# Mock imports for development environment
try:
    import numpy as np
except ImportError:
    # Mock numpy for development
    class MockNumPy:
        uint8 = 'uint8'
        uint16 = 'uint16'
        int16 = 'int16' 
        float32 = 'float32'
        
        def array(self, data, dtype=None):
            return data
            
        def zeros(self, shape, dtype=None):
            if isinstance(shape, (list, tuple)):
                if len(shape) == 2:
                    return [[0 for _ in range(shape[1])] for _ in range(shape[0])]
                elif len(shape) == 3:
                    return [[[0 for _ in range(shape[2])] for _ in range(shape[1])] for _ in range(shape[0])]
            return 0
            
        def ascontiguousarray(self, arr):
            return arr
            
    np = MockNumPy()

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    # Mock GDAL for development
    class MockGDAL:
        GDT_Byte = 1
        GDT_UInt16 = 2
        GDT_Int16 = 3
        GDT_Float32 = 6
        
        GF_Read = 0
        
        class MockDataset:
            def __init__(self):
                self.RasterXSize = 100
                self.RasterYSize = 100
                self.RasterCount = 3
                
            def GetRasterBand(self, band):
                return MockGDAL.MockBand()
                
            def ReadAsArray(self, xoff=0, yoff=0, xsize=None, ysize=None):
                return [[[0 for _ in range(3)] for _ in range(100)] for _ in range(100)]
                
            def GetGeoTransform(self):
                return (0, 1, 0, 0, 0, -1)
                
            def GetProjection(self):
                return ""
                
        class MockBand:
            def __init__(self):
                self.DataType = MockGDAL.GDT_Byte
                
            def ReadAsArray(self, xoff=0, yoff=0, win_xsize=None, win_ysize=None):
                return [[0 for _ in range(100)] for _ in range(100)]
                
            def ReadRaster(self, xoff, yoff, xsize, ysize, buf_xsize=None, buf_ysize=None, buf_type=None):
                return b'\x00' * (xsize * ysize)
                
        def Open(self, filename):
            if os.path.exists(filename):
                return self.MockDataset()
            return None
            
        def UseExceptions(self):
            pass
            
    gdal = MockGDAL()


def safe_file_path(filepath: str) -> str:
    """
    安全处理文件路径，确保正确处理中文字符和Unicode编码
    
    Args:
        filepath: 原始文件路径
        
    Returns:
        处理后的安全文件路径
    """
    if not isinstance(filepath, str):
        return str(filepath)
        
    # 处理Unicode编码问题
    try:
        # 尝试使用UTF-8编码
        if isinstance(filepath.encode('utf-8'), bytes):
            return filepath
    except UnicodeEncodeError:
        # 如果UTF-8编码失败，尝试其他编码
        try:
            # 尝试使用系统默认编码
            return filepath.encode(sys.getdefaultencoding()).decode('utf-8', errors='ignore')
        except (UnicodeDecodeError, UnicodeEncodeError):
            # 最后的备选方案，移除非ASCII字符
            return ''.join(char for char in filepath if ord(char) < 128)
    
    return filepath


def load_raster_image(filepath: str, bands: Optional[Union[int, Tuple[int, ...]]] = None) -> Optional[Any]:
    """
    加载栅格图像数据 (LoadRasterImage函数的Python实现)
    
    Args:
        filepath: 图像文件路径，支持中文路径
        bands: 要读取的波段，None表示读取所有波段
        
    Returns:
        GDAL数据集对象，失败时返回None
        
    Raises:
        RuntimeError: 当文件无法打开时
        UnicodeError: 当路径编码有问题时
    """
    try:
        # 安全处理文件路径
        safe_path = safe_file_path(filepath)
        
        # 检查文件是否存在
        if not os.path.exists(safe_path):
            raise FileNotFoundError(f"文件不存在: {safe_path}")
            
        # 打开GDAL数据集
        dataset = gdal.Open(safe_path)
        if dataset is None:
            raise RuntimeError(f"无法打开栅格文件: {safe_path}")
            
        return dataset
        
    except Exception as e:
        warnings.warn(f"加载栅格图像失败: {str(e)}")
        return None


def gdal_to_image_array(dataset: Any, bands: Optional[Union[int, Tuple[int, ...]]] = None, 
                       convert_bgr_to_rgb: bool = True) -> Optional[Any]:
    """
    将GDAL数据集转换为图像数组 (GDALToHBITMAP函数的Python等价实现)
    
    Args:
        dataset: GDAL数据集对象
        bands: 要转换的波段编号，None表示使用前3个波段
        convert_bgr_to_rgb: 是否将BGR转换为RGB
        
    Returns:
        图像数组 (numpy array或模拟数组)，失败时返回None
    """
    if dataset is None:
        return None
        
    try:
        # 获取数据集基本信息
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        band_count = dataset.RasterCount
        
        # 确定要读取的波段
        if bands is None:
            if band_count >= 3:
                bands = (1, 2, 3)  # RGB
            elif band_count == 1:
                bands = (1,)  # 灰度图
            else:
                bands = tuple(range(1, min(band_count + 1, 4)))  # 最多4个波段
        elif isinstance(bands, int):
            bands = (bands,)
            
        # 读取波段数据
        band_arrays = []
        for band_num in bands:
            if band_num > band_count:
                warnings.warn(f"波段 {band_num} 超出范围，跳过")
                continue
                
            band = dataset.GetRasterBand(band_num)
            if band is None:
                warnings.warn(f"无法获取波段 {band_num}")
                continue
                
            # 使用正确的RasterIO参数读取数据
            band_data = band.ReadAsArray(0, 0, width, height)
            if band_data is not None:
                band_arrays.append(band_data)
                
        if not band_arrays:
            return None
            
        # 组合波段数据
        if len(band_arrays) == 1:
            # 灰度图像
            image_array = band_arrays[0]
        else:
            # 多波段图像，堆叠为RGB/RGBA
            try:
                image_array = np.dstack(band_arrays)
            except:
                # 备选方案：手动堆叠
                height, width = len(band_arrays[0]), len(band_arrays[0][0])
                channels = len(band_arrays)
                image_array = np.zeros((height, width, channels), dtype=np.uint8)
                
                for i, band_data in enumerate(band_arrays):
                    for y in range(height):
                        for x in range(width):
                            image_array[y][x][i] = band_data[y][x]
                            
        # BGR到RGB转换
        if convert_bgr_to_rgb and len(band_arrays) >= 3:
            try:
                # 交换红蓝通道
                if hasattr(image_array, 'shape') and len(image_array.shape) == 3:
                    # numpy方式
                    image_array = image_array[:, :, [2, 1, 0]]
                else:
                    # 手动方式
                    height, width = len(image_array), len(image_array[0])
                    for y in range(height):
                        for x in range(width):
                            if len(image_array[y][x]) >= 3:
                                # 交换R和B通道
                                r, g, b = image_array[y][x][0], image_array[y][x][1], image_array[y][x][2]
                                image_array[y][x][0] = b
                                image_array[y][x][2] = r
            except Exception as e:
                warnings.warn(f"BGR到RGB转换失败: {e}")
                
        return image_array
        
    except Exception as e:
        warnings.warn(f"数据集转换失败: {str(e)}")
        return None


def normalize_image_data(image_array: Any, data_type: str = 'uint8') -> Any:
    """
    标准化图像数据到指定的数据类型范围
    
    Args:
        image_array: 输入图像数组
        data_type: 目标数据类型 ('uint8', 'uint16', 'float32')
        
    Returns:
        标准化后的图像数组
    """
    if image_array is None:
        return None
        
    try:
        # 获取数据的最小值和最大值
        if hasattr(image_array, 'min') and hasattr(image_array, 'max'):
            # numpy方式
            min_val = image_array.min()
            max_val = image_array.max()
        else:
            # 手动计算
            flat_data = []
            if hasattr(image_array, 'shape'):
                shape = image_array.shape
                if len(shape) == 2:
                    for row in image_array:
                        flat_data.extend(row)
                elif len(shape) == 3:
                    for row in image_array:
                        for pixel in row:
                            if isinstance(pixel, (list, tuple)):
                                flat_data.extend(pixel)
                            else:
                                flat_data.append(pixel)
            min_val = min(flat_data) if flat_data else 0
            max_val = max(flat_data) if flat_data else 1
            
        # 避免除零错误
        if max_val == min_val:
            return image_array
            
        # 根据数据类型进行标准化
        if data_type == 'uint8':
            scale_factor = 255.0 / (max_val - min_val)
            try:
                return ((image_array - min_val) * scale_factor).astype(np.uint8)
            except:
                # 手动标准化
                return _manual_normalize(image_array, min_val, max_val, 255)
        elif data_type == 'uint16':
            scale_factor = 65535.0 / (max_val - min_val)
            try:
                return ((image_array - min_val) * scale_factor).astype(np.uint16)
            except:
                return _manual_normalize(image_array, min_val, max_val, 65535)
        elif data_type == 'float32':
            try:
                return ((image_array - min_val) / (max_val - min_val)).astype(np.float32)
            except:
                return _manual_normalize(image_array, min_val, max_val, 1.0)
                
    except Exception as e:
        warnings.warn(f"图像标准化失败: {e}")
        return image_array
        
    return image_array


def _manual_normalize(image_array: Any, min_val: float, max_val: float, target_max: float) -> Any:
    """
    手动标准化图像数据（备选实现）
    """
    try:
        scale_factor = target_max / (max_val - min_val)
        
        if hasattr(image_array, 'shape'):
            shape = image_array.shape
            if len(shape) == 2:
                # 二维数组
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        image_array[i][j] = int((image_array[i][j] - min_val) * scale_factor)
            elif len(shape) == 3:
                # 三维数组
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        for k in range(shape[2]):
                            image_array[i][j][k] = int((image_array[i][j][k] - min_val) * scale_factor)
                            
        return image_array
    except Exception:
        return image_array


def get_image_info(dataset: Any) -> dict:
    """
    获取图像的基本信息
    
    Args:
        dataset: GDAL数据集对象
        
    Returns:
        包含图像信息的字典
    """
    if dataset is None:
        return {}
        
    try:
        info = {
            'width': dataset.RasterXSize,
            'height': dataset.RasterYSize,
            'bands': dataset.RasterCount,
            'geotransform': dataset.GetGeoTransform(),
            'projection': dataset.GetProjection()
        }
        
        # 获取波段信息
        band_info = []
        for i in range(1, dataset.RasterCount + 1):
            band = dataset.GetRasterBand(i)
            if band:
                band_info.append({
                    'band': i,
                    'data_type': band.DataType,
                    'no_data_value': band.GetNoDataValue()
                })
        info['band_info'] = band_info
        
        return info
        
    except Exception as e:
        warnings.warn(f"获取图像信息失败: {e}")
        return {}


class RasterImageProcessor:
    """
    栅格图像处理类，提供完整的图像加载、转换和管理功能
    """
    
    def __init__(self):
        self.dataset = None
        self.image_array = None
        self.image_info = {}
        
    def __del__(self):
        """释放资源"""
        self.cleanup()
        
    def cleanup(self):
        """清理资源，防止内存泄漏"""
        if self.dataset is not None:
            self.dataset = None
        self.image_array = None
        self.image_info = {}
        
    def load_image(self, filepath: str, bands: Optional[Union[int, Tuple[int, ...]]] = None) -> bool:
        """
        加载图像
        
        Args:
            filepath: 图像文件路径
            bands: 要加载的波段
            
        Returns:
            加载成功返回True，失败返回False
        """
        try:
            # 清理之前的资源
            self.cleanup()
            
            # 加载数据集
            self.dataset = load_raster_image(filepath, bands)
            if self.dataset is None:
                return False
                
            # 获取图像信息
            self.image_info = get_image_info(self.dataset)
            
            # 转换为图像数组
            self.image_array = gdal_to_image_array(self.dataset, bands)
            
            return self.image_array is not None
            
        except Exception as e:
            warnings.warn(f"加载图像失败: {e}")
            self.cleanup()
            return False
            
    def get_image_array(self, normalize: bool = True, data_type: str = 'uint8') -> Optional[Any]:
        """
        获取图像数组
        
        Args:
            normalize: 是否标准化数据
            data_type: 数据类型
            
        Returns:
            图像数组
        """
        if self.image_array is None:
            return None
            
        if normalize:
            return normalize_image_data(self.image_array, data_type)
        else:
            return self.image_array
            
    def get_info(self) -> dict:
        """获取图像信息"""
        return self.image_info.copy()
        
    def is_loaded(self) -> bool:
        """检查图像是否已加载"""
        return self.dataset is not None and self.image_array is not None


# 便捷函数
def quick_load_image(filepath: str, bands: Optional[Union[int, Tuple[int, ...]]] = None, 
                    normalize: bool = True) -> Optional[Any]:
    """
    快速加载图像的便捷函数
    
    Args:
        filepath: 图像文件路径
        bands: 要加载的波段
        normalize: 是否标准化数据
        
    Returns:
        图像数组，失败时返回None
    """
    processor = RasterImageProcessor()
    try:
        if processor.load_image(filepath, bands):
            return processor.get_image_array(normalize=normalize)
        return None
    finally:
        processor.cleanup()


if __name__ == "__main__":
    # 简单测试
    print("Image utilities module loaded successfully")
    
    # 测试路径处理
    test_path = "测试文件.tif"
    safe_path = safe_file_path(test_path)
    print(f"Safe path: {safe_path}")
    
    # 测试处理器
    processor = RasterImageProcessor()
    print(f"Processor created: {processor.is_loaded()}")