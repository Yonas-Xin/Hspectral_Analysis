"""该文档用于将遥感数据读取为图片格式"""
import numpy as np
from osgeo import gdal, ogr, osr
import matplotlib.pyplot as plt
import cv2
gdal.UseExceptions()
GDAL2NP_TYPE = { # GDAL数据类型与numpy数据类型的映射
    gdal.GDT_Byte: ('uint8', np.uint8),
    gdal.GDT_UInt16: ('uint16', np.uint16),
    gdal.GDT_Int16: ('int16', np.int16),
    gdal.GDT_UInt32: ('uint32', np.uint32),
    gdal.GDT_Int32: ('int32', np.int32),
    gdal.GDT_Float32: ('float32', np.float32),
    gdal.GDT_Float64: ('float64', np.float64)
}
NP2GDAL_TYPE = {
    np.dtype('uint8'): gdal.GDT_Byte,
    np.dtype('uint16'): gdal.GDT_UInt16,
    np.dtype('int16'): gdal.GDT_Int16,
    np.dtype('uint32'): gdal.GDT_UInt32,
    np.dtype('int32'): gdal.GDT_Int32,
    np.dtype('float32'): gdal.GDT_Float32,
    np.dtype('float64'): gdal.GDT_Float64
}

class Gdal_Tool(object):
    def __init__(self, input_tif):
        self.input_tif = input_tif
        dataset = gdal.Open(input_tif)
        band = dataset.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()
        self.bands = dataset.RasterCount
        self.rows, self.cols = dataset.RasterYSize, dataset.RasterXSize
        self.geotransform = dataset.GetGeoTransform()
        self.projection = dataset.GetProjection()
        dataset = None

    def read_tif_to_image(self, band, stretch = "Linear", to_gray = True, to_int = True):
        """stretch: Linear or Linear_2%"""
        if stretch == "Linear":
            stretch_func = linear_stretch
        elif stretch == "Linear_2%":
            stretch_func = linear_2pct_stretch
        else: raise ValueError("不支持该拉伸方法")

        dataset = gdal.Open(self.input_tif)
        if isinstance(band, tuple):
            if len(band) != 1 and len(band) != 3:
                raise ValueError("波段数量只支持1或者3!")
        else: band = (band, ) # 如果是int，则转为tuple
        data = np.stack([dataset.GetRasterBand(i).ReadAsArray().astype(float) for i in band], axis=0)
        if self.nodata is not None: # 设置nan数据
            data[data == self.nodata] = np.nan
        image = np.stack([stretch_func(data[i]) for i in range(len(band))], axis=0).transpose(1,2,0).squeeze() # float32
        if to_int:
            image = (image*255).astype(np.uint8)
        if to_gray == True and len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) # 转灰度图
        return image

    
    def save_tif(self, out_path, data, factor = None):
        '''
        用于保存提取的构造结果（二值图）
        data (np.ndarray): 三维数组（波段,行,列）
        '''
        if len(data.shape) == 2:
            rows, cols = data.shape
            bands = 1
            data = data.reshape((1, rows, cols))  # 转换为三维
        elif len(data.shape) == 3:
            bands, rows, cols = data.shape
        else:
            raise ValueError("输入数据必须是二维或三维数组")
        if (self.rows != rows or self.cols != cols) and factor is None:
            raise ValueError("输入数据形状与影像大小不符")
        try:
            dtype = NP2GDAL_TYPE[data.dtype]
        except KeyError:
            raise ValueError(f"不支持的数据类型: {data.dtype}. 支持的类型包括: {list(NP2GDAL_TYPE.keys())}")
        
        driver = gdal.GetDriverByName("GTiff")
        dataset = driver.Create(out_path, cols, rows, bands, dtype)
        if dataset is None:
            raise IOError(f"无法创建文件 {out_path}")
        # 设置地理变换和投影
        if self.geotransform is not None and self.projection is not None:
            if factor is not None:
                gt = list(self.geotransform)
                gt[1] *= factor  # 像元宽度
                gt[5] *= factor  # 像元高度
                geotransform = tuple(gt)
            dataset.SetGeoTransform(geotransform)
            dataset.SetProjection(self.projection)
        # 写入数据
        for i in range(bands):
            band = dataset.GetRasterBand(i + 1)
            band.WriteArray(data[i,:,:])
        # 释放资源
        dataset.FlushCache()
        dataset = None
        return out_path
    
    def skeleton_to_shp_from_raster(self, binary_img, shp_path):
        """
        将骨架二值图转为线状Shapefile（自动读取原始影像的空间参考）
        
        binary_img: 二值图(0/1 或 0/255)，shape应与原始影像一致
        shp_path: 输出shp文件路径
        raster_path: 原始影像路径（例如 ori.dat）
        """
        geotransform = self.geotransform
        projection = self.projection

        origin_x = geotransform[0]
        pixel_size_x = geotransform[1]
        origin_y = geotransform[3]
        pixel_size_y = geotransform[5]  # 注意可能为负

        # 转成0/255格式
        if binary_img.max() <= 1:
            img_255 = (binary_img * 255).astype(np.uint8)
        else:
            img_255 = binary_img.astype(np.uint8)

        # 查找骨架线的轮廓
        contours, _ = cv2.findContours(img_255, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # 创建Shapefile
        driver = ogr.GetDriverByName("ESRI Shapefile")
        # driver.DeleteDataSource(shp_path)
        datasource = driver.CreateDataSource(shp_path)

        srs = osr.SpatialReference()
        if projection:  # 如果原影像有投影
            srs.ImportFromWkt(projection)
        else:
            srs.ImportFromEPSG(4326)  # 默认WGS84

        layer = datasource.CreateLayer("skeleton", srs, ogr.wkbLineString)

        # 添加属性字段
        field_defn = ogr.FieldDefn("id", ogr.OFTInteger)
        layer.CreateField(field_defn)

        # 将骨架转换为LineString
        for idx, cnt in enumerate(contours):
            if len(cnt) < 2:
                continue
            line = ogr.Geometry(ogr.wkbLineString)
            for pt in cnt:
                px, py = pt[0]
                # 转换到地理坐标
                x_geo = origin_x + px * pixel_size_x
                y_geo = origin_y + py * pixel_size_y
                line.AddPoint(x_geo, y_geo)

            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("id", idx + 1)
            feature.SetGeometry(line)
            layer.CreateFeature(feature)
            feature = None

        datasource = None
        print(f"✅ Shapefile 已保存到: {shp_path}")

def linear_2pct_stretch(arr):
    """
    对二维数组进行 linear 2% 拉伸，忽略 NaN 值
    输入: arr [rows, cols]
    输出: [rows, cols] (float，范围0~1)
    """
    # 展平数据并去掉 NaN
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)  # 全是NaN时返回0矩阵
    p2 = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    if p98 == p2:
        return np.zeros_like(arr)
    stretched = (arr - p2) / (p98 - p2)
    stretched = np.clip(stretched, 0, 1)
    return stretched

def linear_stretch(arr):
    # 展平数据并去掉 NaN
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return np.zeros_like(arr)  # 全是NaN时返回0矩阵
    
    # 计算有效数据的最小最大值
    min_val = np.min(valid)
    max_val = np.max(valid)
    if min_val == max_val:
        return np.zeros_like(arr)
    stretched = (arr - min_val) / (max_val - min_val)
    stretched = np.clip(stretched, 0, 1)
    return stretched


if '__main__' == __name__:
    # 测试
    input = r'C:\Users\85002\Desktop\TempDIR\out2.dat'
    input1 = r'C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\whole_area_138.dat'
    gt = Gdal_Tool(input1)
    image = gt.read_tif_to_image(1)
    plt.imshow(image)
    plt.show()