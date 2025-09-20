import os.path
try:
    from osgeo import gdal,ogr,osr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
import sys
import random

nodata_value = 0
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
def write_data_to_tif(output_file, data, geotransform, projection, nodata_value=nodata_value):
    """
    将数组数据写入GeoTIFF文件
    
    参数:
        output_file (str): 输出文件路径
        data (np.ndarray): 三维数组（波段,行,列）
        geotransform (tuple): 6参数地理变换
        projection (str): WKT格式的坐标参考系统
        nodata_value (int/float): NoData值, 默认0
    
    异常:
        IOError: 当文件创建失败时
    """
    # 处理二维数组情况（单波段）
    if len(data.shape) == 2:
        rows, cols = data.shape
        bands = 1
        data = data.reshape((1, rows, cols))  # 转换为三维
    elif len(data.shape) == 3:
        bands, rows, cols = data.shape
    else:
        raise ValueError("输入数据必须是二维或三维数组")
    
    try:
        dtype = NP2GDAL_TYPE[data.dtype]
    except KeyError:
        raise ValueError(f"不支持的数据类型: {data.dtype}. 支持的类型包括: {list(NP2GDAL_TYPE.keys())}")
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(output_file, cols, rows, bands, dtype)
    if dataset is None:
        raise IOError(f"无法创建文件 {output_file}")
    # 设置地理变换和投影
    if geotransform is not None and projection is not None:
        dataset.SetGeoTransform(geotransform)
        dataset.SetProjection(projection)
    # 写入数据
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data[i,:,:])
        band.SetNoDataValue(nodata_value)  # 设置 NoData 值
    # 释放资源
    dataset.FlushCache()
    dataset = None
    return output_file

def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据(形状为 bands x rows x cols)'''
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) * 1e-4
    return dataset

def read_ori_tif(tif_path):
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    return dataset

def crop_image_by_mask(data, mask, geotransform, projection, filepath, block_size=30, name="Block_"):
    """
    根据 mask 的类别, 裁剪影像为 30x30 的小块
    :param data: 输入影像, (C, H, W)
    :param mask: 二维 mask, 形状为 (rows, cols), 背景为0, 其他为类别
    :param block_size: 每个块的大小, 默认为30
    :return: ndarray,(nums,block_size,block_size,bands)
    """
    bands, rows, cols = data.shape
    if block_size%2 == 0:#如果block_size是一偶数, 以像素点为中心，左上角区域比右下角区域少一
        left_top = int(block_size/2-1)
        right_bottom = int(block_size/2)
    else:
        left_top = int(block_size//2)
        right_bottom = int(block_size//2)
    data = np.pad(data,[(0,0),(left_top,right_bottom),(left_top,right_bottom)],'constant')
    num = 1
    pathlist = []
    add_labels = False
    # 遍历 mask 的每个像素
    if np.max(mask)>1: # 如果大于1说明裁剪的图像有标签
        print('有标签，将额外生成标签至txt文件')
        add_labels = True
    else: print('无标签，生成纯数据地址txt文件')
    for row in tqdm(range(rows),desc='Cropping', total=rows):
        for col in range(cols):
            if mask[row, col] > 0:  # 如果该位置不是背景和噪声
                path = os.path.join(filepath, name + str(num) + ".tif")
                # 计算裁剪图像的左上角坐标（originX, originY）
                originX = geotransform[0] + (col-left_top) * geotransform[1]
                originY = geotransform[3] + (row-left_top) * geotransform[5]
                # 计算新的 GeoTransform
                new_geotransform = (originX, geotransform[1], geotransform[2], originY, geotransform[4], geotransform[5])
                block = data[:, row:row + block_size, col:col + block_size]
                write_data_to_tif(path, block, geotransform=new_geotransform, projection=projection)
                num += 1
                if add_labels:
                    pathlist.append(path + f' {mask[row, col]-1}')
                else: pathlist.append(path)
    dataset_path = os.path.join(filepath, '.datasets.txt')
    write_list_to_txt(pathlist, dataset_path)
    # if np.max(mask)>1: # 如果大于1说明裁剪的图像有标签
    #     label_path = os.path.join(filepath, '.labels.txt')
    #
    #     write_list_to_txt(labels,label_path) # 裁剪的文件夹下创建label txt文件
    #     write_list_to_txt(pathlist, dataset_path) # 裁剪的文件夹下创建dataset txt文件
    #     print(f'样本路径下生成了标签文件：{label_path}')
    #     print(f'样本路径下生成了数据集文件：{dataset_path}')
    # else: # 生成无标签的样本集
    #     dataset_path = os.path.join(filepath, '.datasets.txt')
    #     write_list_to_txt(pathlist, dataset_path) # 裁剪的文件夹下创建dataset txt文件
    #     print(f'样本路径下生成了数据集文件(无标签)：{dataset_path}')

def write_list_to_txt(data, filename):
    with open(filename, 'w') as file:
        for item in data:
            file.write(f"{item}\n")  # 每个元素后加上换行符
        file.flush()

def create_dataset_from_file(filepath, extension = '.tif'):
    images = search_files_in_directory(filepath, extension = extension)
    return images

def search_files_in_directory(directory, extension='.txt'):
    """
    搜索指定文件夹中所有指定后缀名的文件，并返回文件路径列表
    Parameters:
        directory (str): 要搜索的文件夹路径
        extension (str): 文件后缀名，应该以 '.' 开头，例如 '.txt', '.jpg'
    Returns:
        list: 包含所有符合条件的文件路径的列表
    """
    matching_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                matching_files.append(os.path.join(root, file))
    return matching_files

def mask_to_vector_gdal(mask_matrix,geotransform, projection=None, output_shapefile="./Position_mask/test.shp"):
    """
    将二维矩阵mask矩阵转化为矢量点文件
    """
    # 获取矩阵的行和列
    rows, cols = mask_matrix.shape
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if not driver:
        raise RuntimeError("Shapefile driver not available")
    data_source = driver.CreateDataSource(output_shapefile)
    # 创建一个图层，用于存储点（Point）几何
    spatial_ref = osr.SpatialReference()
    if projection:
        spatial_ref.ImportFromWkt(projection) # 定义投影坐标系
    layer = data_source.CreateLayer('points', geom_type=ogr.wkbPoint, srs=spatial_ref)
    field = ogr.FieldDefn('class', ogr.OFTInteger)
    layer.CreateField(field)
    # 遍历矩阵，提取非零值的坐标并创建点特征
    for row in range(rows):
        for col in range(cols):
            value = mask_matrix[row, col]
            if value > 0:  # 非零值表示分类
                # 创建一个点
                geo_x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
                geo_y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
                point = ogr.Geometry(ogr.wkbPoint)
                # if write_value is None:
                #     write_value = int(value-1)
                point.AddPoint(geo_x, geo_y)
                # 创建一个要素（Feature）并设置几何和属性值
                feature = ogr.Feature(layer.GetLayerDefn())
                feature.SetGeometry(point)
                feature.SetField('class', 0)  # 设置分类属性值
                layer.CreateFeature(feature)  # 将特征写入图层
                feature = None  # 清理
    # 关闭数据源，保存Shapefile
    data_source = None
    print(f"{output_shapefile} has been created successfully.")

def mask_to_multivector(mask_matrix, geotransform, projection=None, output_dir="./Position_mask", output_dir_name=None):
    """
    在指定文件夹下创建一个新文件夹保存样本的矢量点文件"""
    labels = np.unique(mask_matrix)
    if output_dir_name is None:
        output_dir_name = 'SAMPLES_DIR'
    OUTPUT_DIR = os.path.join(output_dir, output_dir_name)
    counter = 1
    while os.path.exists(OUTPUT_DIR):
        OUTPUT_DIR = os.path.join(output_dir, f"{output_dir_name}_{counter}")
        counter += 1
    os.makedirs(OUTPUT_DIR)
    for label in labels:
        if label > 0:
            input = np.zeros_like(mask_matrix, dtype=np.int16)
            time.sleep(1) # 确保间隔一秒创建一个文件
            current_time = datetime.now().strftime("%Y%m%d%H%M%S")
            output_shapefile = os.path.join(OUTPUT_DIR, f"{current_time}_sample.shp")
            input[mask_matrix == label] = label
            mask_to_vector_gdal(input, geotransform, projection, output_shapefile)

def vector_to_mask(shapefile, geotransform, rows, cols, value=None):
    """
    将矢量点文件转化为二维矩阵（mask矩阵）
    mask属性值从1开始，背景为0
    """
    mask_matrix = np.zeros((rows, cols), dtype=int)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = ogr.Open(shapefile)
    if not data_source:
        raise RuntimeError(f"Failed to open shapefile: {shapefile}")
    layer = data_source.GetLayer()
    field_idx = layer.FindFieldIndex('class', 1)
    if field_idx != -1:
        has_field_class = True
    else: has_field_class = False
    spatial_ref = layer.GetSpatialRef()
    if spatial_ref is None:
        raise RuntimeError("No spatial reference found in shapefile")
    # 处理每个要素（点）
    for feature in layer:
        # 获取点的几何体
        geometry = feature.GetGeometryRef()
        if geometry.GetGeometryType() != ogr.wkbPoint:
            continue  # 只处理点类型的几何
        geo_x, geo_y = geometry.GetX(), geometry.GetY()
        col = int((geo_x - geotransform[0]) / geotransform[1])  # 计算列索引
        row = int((geo_y - geotransform[3]) / geotransform[5])  # 计算行索引
        if 0 <= row < rows and 0 <= col < cols:        # 确保索引在矩阵范围内
            if value is not None:
                mask_matrix[row, col] = value
            # else:mask_matrix[row, col] = feature.GetField('class') + 1  # 如果没有指定值，则使用字段“class”值
            else:
                if has_field_class:
                    class_value = feature.GetField('class')
                    mask_matrix[row, col] = class_value + 1
                else: mask_matrix[row, col] = 1 # 默认点位置取值为1
    data_source = None
    return mask_matrix

def mutivetor_to_mask(shapefile_dir, geotransform, rows, cols):
    """
    将指定文件夹下的所有矢量点文件转化为二维矩阵（mask矩阵）
    """
    mask_matrix = np.zeros((rows, cols), dtype=int)
    shapefiles = search_files_in_directory(shapefile_dir, extension='.shp')
    if not shapefiles:
        raise RuntimeError(f"No shapefiles found in directory: {shapefile_dir}")
    for idx, shapefile in enumerate(shapefiles):
        temp_mask = vector_to_mask(shapefile, geotransform, rows, cols, value=idx+1)
        # 检查是否有重叠（即 mask_matrix 和 temp_mask 在相同位置都有非零值）
        overlap = np.logical_and(mask_matrix > 0, temp_mask > 0)
        if np.any(overlap):
            raise RuntimeError(
                f"There is an overlap in the sample points, please check the file! "
            )
        mask_matrix += temp_mask
    return mask_matrix

def point_value_merge(shapefile, value:list):
    """
    要素属性修正
    """
    base_value = value[0]
    merge_value = value[1:]
    data_source = ogr.Open(shapefile, 1)
    if not data_source:
        raise RuntimeError(f"Failed to open shapefile: {shapefile}")
    try:
        layer = data_source.GetLayer()
        modified_count = 0
        all_count = 0

        # 遍历所有要素
        for feature in layer:
            # 验证几何类型为点
            all_count += 1
            geometry = feature.GetGeometryRef()
            if geometry and geometry.GetGeometryType() != ogr.wkbPoint:
                continue

            # 获取字段值
            field_value = feature.GetField('class')

            # 执行属性修改
            if field_value in merge_value:
                feature.SetField('class', base_value)
                layer.SetFeature(feature)  # 提交修改
                modified_count += 1

        print(f'要素属性修正完成，共修改 {modified_count} 个点要素，文件总要素{all_count}')

    finally:
        # 确保资源释放
        data_source.Destroy()
        data_source = None

def crop_image_by_mask_block(image_file, out_filepath, sampling_position, image_block=256, block_size=30, name="Block_"):
    '''裁剪样本，适合无法一次加载到内存的大影像'''
    dataset = gdal.Open(image_file)
    geotransform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    rows, cols = dataset.RasterYSize, dataset.RasterXSize
    if block_size % 2 == 0:  # 如果block_size是一偶数，以像素点为中心，左上角区域比右下角区域少一
        left_top = int(block_size / 2 - 1)
        right_bottom = int(block_size / 2)
    else:
        left_top = int(block_size // 2)
        right_bottom = int(block_size // 2)
    num = 1
    pathlist = []
    add_labels = False
    if np.max(sampling_position) > 1:  # 如果大于1说明裁剪的图像有标签
        print('有标签，将额外生成标签至txt文件')
        add_labels = True
    else:
        print('无标签，生成纯数据地址txt文件')
    for i in range(0, rows, image_block):
        for j in range(0, cols, image_block):
            # 计算当前块的实际高度和宽度（避免越界）
            actual_rows = min(image_block+block_size-1, rows - i)#实际高
            actual_cols = min(image_block+block_size-1, cols - j)#实际宽
            if (j-left_top)<0:
                xoff=0
                actual_cols-=left_top
                left_pad=left_top
            else:
                xoff=j-left_top
                left_pad=0
            if (i-left_top)<0:
                yoff=0
                actual_rows-=left_top
                top_pad=left_top
            else:
                yoff=i-left_top
                top_pad=0
            if actual_cols==(cols - j):
                actual_cols += left_top
                right_pad = right_bottom
            else:right_pad = 0
            if actual_rows==(rows - i):
                actual_rows += left_top
                bottom_pad = right_bottom
            else:bottom_pad = 0
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
            block_data = dataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=actual_cols, ysize=actual_rows)
            if block_data.dtype == np.int16:
                block_data = block_data.astype(np.float32) * 1e-4
            block_data = np.pad(block_data,[(0, 0), (top_pad, bottom_pad), (left_pad, right_pad)], 'constant')

            row_block = min(image_block, rows - i) # 记录真实窗口大小
            col_block = min(image_block, cols - j)
            block_sampling_mask = sampling_position[i:i + row_block, j:j + col_block]
            # block_sampling_mask = np.pad(block_sampling_mask,[(left_top, right_bottom), (left_top, right_bottom)], 'constant')
            # show_img(block_data)
            _, block_rows, block_cols = block_data.shape
            oringinx = geotransform[0]+j*geotransform[1]
            oringiny = geotransform[3]+i*geotransform[5]
            if np.all(block_sampling_mask==0):
                continue
            pbar = tqdm(total=int(np.sum(block_sampling_mask > 0))) # 进度条
            for row in range(row_block):
                for col in range(col_block):
                    if block_sampling_mask[row, col] > 0:  # 如果该位置不是背景和噪声
                        oringinX = (col-left_top)*geotransform[1]+oringinx
                        oringinY = (row-left_top)*geotransform[5]+oringiny
                        new_geotransform = (oringinX, geotransform[1], geotransform[2], oringinY, geotransform[4], geotransform[5])
                        path = os.path.join(out_filepath, name + f'{block_size}_{block_size}_{num}.tif')
                        block = block_data[:, row:row + block_size, col:col+block_size]
                        write_data_to_tif(path, block, geotransform=new_geotransform, projection=projection)
                        if add_labels:
                            pathlist.append(path + f' {block_sampling_mask[row, col] - 1}')
                        else:
                            pathlist.append(path)
                        num+=1
                        pbar.update(1)
    dataset_path = os.path.join(out_filepath, '.datasets.txt')
    write_list_to_txt(pathlist, dataset_path)
    print('样本裁剪完成')

def face_vector_to_mask(shp_path, geotransform, projection, rows, cols, str='class', output_path=None):
    """
    将面Shapefile转换为分类栅格数组

    参数:
        shp_path: str - Shapefile路径(.shp)
        geotransform: tuple - GDAL地理变换参数(6个值)
        projection: str - WKT格式的坐标参考系统
        rows: int - 输出栅格行数-高
        cols: int - 输出栅格列数-宽

    返回:
        numpy.ndarray - 二维数组，被面覆盖的像素为class值(0,1,2...)，其余为-1

    异常:
        ValueError - 当class字段值不是整数时
        RuntimeError - GDAL操作失败时
    """
    shp_ds = ogr.Open(shp_path)
    shp_layer = shp_ds.GetLayer()
    layer_defn = shp_layer.GetLayerDefn()
    
    # 检查class字段是否存在且为整数类型
    class_field_idx = layer_defn.GetFieldIndex(str)
    if class_field_idx == -1:
        raise RuntimeError(f"Shapefile中缺少'{str}'字段")
    
    field_type = layer_defn.GetFieldDefn(class_field_idx).GetType()
    if field_type not in (ogr.OFTInteger, ogr.OFTInteger64):
        raise ValueError(f"'{str}'字段必须是整型")
    # 创建内存栅格
    driver = gdal.GetDriverByName('MEM')
    raster_ds = driver.Create(
        '', 
        cols, 
        rows, 
        1, 
        gdal.GDT_Int16
    )
    raster_ds.SetGeoTransform(geotransform)
    raster_ds.SetProjection(projection)

    band = raster_ds.GetRasterBand(1)
    band.WriteArray(np.full((rows, cols), -1)) # 值为-1
    band.SetNoDataValue(-1)
    options = [
        f'ATTRIBUTE={str}',  # 使用class字段作为像素值
        'ALL_TOUCHED=TRUE'  # 部分覆盖的像素也计入
    ]
    gdal.RasterizeLayer(
        raster_ds, 
        [1],  # 波段号
        shp_layer,
        options=options
    )
    result = band.ReadAsArray()
    if output_path:
        out_driver = gdal.GetDriverByName('GTiff')
        out_ds = out_driver.CreateCopy(output_path, raster_ds)
        out_ds = None  # 关闭文件
    raster_ds = None
    shp_ds = None
    return result

def clip_by_shp(out_dir, sr_img, point_shp, block_size=30, out_tif_name='img', fill_value=0, value=None):
    """
    根据点Shapefile从影像中裁剪指定大小的图像块
    
    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象
        point_shp (str): 点要素Shapefile路径
        block_size (int): 裁剪块大小（像素），默认30
        out_tif_name (str): 输出文件名前缀，默认'img'
        fill_value (int/float): 边缘填充值，默认0
        value (int): 为输出文件名添加的标签值, 默认None
    
    返回:
        list: 生成的图像路径列表, 格式为["path1.tif label1", "path2.tif label2", ...]
    
    异常:
        RuntimeError: 当无法打开输入文件时
        TypeError: 当sr_img参数类型无效时
    """
    # 计算中心偏移
    if block_size % 2 == 0:
        left_top = block_size // 2 - 1
        right_bottom = block_size // 2
    else:
        left_top = right_bottom = block_size // 2

    # 读取原始影像
    need_close = False
    if isinstance(sr_img, str):
        im_dataset = gdal.Open(sr_img)
        if im_dataset is None:
            raise RuntimeError(f"无法打开影像文件: {sr_img}")
        need_close = True
    elif isinstance(sr_img, gdal.Dataset):
        im_dataset = sr_img
    else:
        raise TypeError("sr_img必须是文件路径字符串或GDAL数据集对象")
    
    im_geotrans = im_dataset.GetGeoTransform()
    im_proj = im_dataset.GetProjection()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    im_bands = im_dataset.RasterCount

    band = im_dataset.GetRasterBand(1)
    data_type = band.DataType # 获取数据类型
    dtype_name, numpy_dtype = GDAL2NP_TYPE.get(data_type, ('unknown', None)) # 确定numpy数据类型
    if dtype_name == 'unknown':
        raise ValueError(f"不支持的GDAL数据类型: {data_type}")
    
    # 读取样本点
    shp_dataset = ogr.Open(point_shp)
    if shp_dataset is None:
        raise RuntimeError(f"无法打开矢量文件: {point_shp}")
    
    layer = shp_dataset.GetLayer()
    count = layer.GetFeatureCount()
    pbar = tqdm(total=count) # 进度条
    idx = 0
    out_dataset = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        geoX, geoY = geom.GetX(), geom.GetY()
        
        # 转换坐标到像素位置
        x = int((geoX - im_geotrans[0]) / im_geotrans[1])
        y = int((geoY - im_geotrans[3]) / im_geotrans[5])
        
        # 计算裁剪窗口
        x_start = x - left_top
        y_start = y - left_top
        x_end = x + right_bottom + 1
        y_end = y + right_bottom + 1
        
        # 计算实际可读取范围
        read_x = max(0, x_start)
        read_y = max(0, y_start)
        read_width = min(x_end, im_width) - read_x
        read_height = min(y_end, im_height) - read_y
        
        # 如果有有效区域可读取
        if read_width > 0 and read_height > 0: # 在影像范围内才进行裁剪
            if im_bands > 1:# 创建填充数组
                full_data = np.full((im_bands, block_size, block_size), fill_value, dtype=numpy_dtype)
            else:
                full_data = np.full((block_size, block_size), fill_value, dtype=numpy_dtype)
            if read_width > 0 and read_height > 0:  
                # 读取实际数据
                if im_bands > 1:
                    data = im_dataset.ReadAsArray(read_x, read_y, read_width, read_height)
                    # 计算在填充数组中的位置
                    offset_x = read_x - x_start
                    offset_y = read_y - y_start
                    # 将数据放入填充数组
                    full_data[:, offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
                else:
                    data = im_dataset.GetRasterBand(1).ReadAsArray(read_x, read_y, read_width, read_height)
                    if data.dtype == np.int16:
                        data = data.astype(np.float32) * 1e-4 # 如何data是int类型, 进行放缩并转化为float32类型
                    offset_x = read_x - x_start
                    offset_y = read_y - y_start
                    full_data[offset_y:offset_y+read_height, offset_x:offset_x+read_width] = data
        
            # 计算新的地理变换
            new_geotrans = list(im_geotrans)
            new_geotrans[0] = im_geotrans[0] + x_start * im_geotrans[1]
            new_geotrans[3] = im_geotrans[3] + y_start * im_geotrans[5]
            
            # 保存结果
            idx += 1
            out_path = os.path.join(out_dir, f"{out_tif_name}_{idx}.tif")
            write_data_to_tif(out_path, full_data, new_geotrans, im_proj)
            if value is not None:
                out_dataset.append(f"{out_path} {value}")
        pbar.update(1)
        
        # print(f"生成: {out_path} (有效区域: {read_width}x{read_height})")
    if idx == 0:
        raise RuntimeError(f'Did not find any valid points in the shapefile: {point_shp}, please check that the image matches the vector range')
    shp_dataset = None
    pbar.close()
    if need_close:
        im_dataset = None
    return out_dataset

def clip_by_multishp(out_dir, sr_img, point_shp_dir, block_size=30, out_tif_name='img', fill_value=0):
    """
    批量处理目录下多个Shapefile的裁剪任务, 并自动生成记录样本块与标签的数据集
    
    参数:
        out_dir (str): 输出目录路径
        sr_img (str/gdal.Dataset): 输入影像路径或已打开的GDAL数据集对象  
        point_shp_dir (str): 包含点Shapefiles的目录路径或者是一个Shapefile文件路径
        block_size (int): 裁剪块大小（像素）, 默认30
        out_tif_name (str): 输出文件名前缀, 默认'img'
        fill_value (int/float): 边缘填充值, 默认0
    
    返回:
        None
    
    异常:
        RuntimeError: 当目录中没有Shapefile或裁剪失败时
    """
    if os.path.isdir(point_shp_dir):
        point_shp_files = search_files_in_directory(point_shp_dir, extension='.shp')
        if not point_shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {point_shp_dir}')
        all_out_datasets = []
        for idx, point_shp in enumerate(point_shp_files):
            out_dataset = clip_by_shp(out_dir, sr_img, point_shp, block_size, out_tif_name=f'{out_tif_name}_label{idx}', fill_value=fill_value, value=idx)
            all_out_datasets.extend(out_dataset)
        if not all_out_datasets:
            pass
        else:
            dataset_path = os.path.join(out_dir, '.datasets.txt')
            write_list_to_txt(all_out_datasets, dataset_path)
            print(f'dataset file saved to: {dataset_path}')
    elif os.path.isfile(point_shp_dir):
        print('single shape file dont need to write dataset file')
        out_dataset = clip_by_shp(out_dir, sr_img, point_shp, block_size, out_tif_name=out_tif_name, fill_value=fill_value)
    else:
        raise RuntimeError(f'Invalid point_shp_dir: {point_shp_dir}, it should be a directory or a shapefile path')
    

def batch_raster_to_vector(tif_dir, shp_img_path, extension='.tif', dict=None, delete_value=0, if_smooth=False):
    """
    批量栅格转矢量, code by why
    :param tif_dir: 输入的需要处理的栅格文件夹
    :param shp_img_path: 输出的矢量路径
    :param extension: 栅格后缀
    :param dict: 类型字典, 如{1: "变质岩", 2: "沉积岩", ...}
    :param delete_value: 需要删除的背景值, 默认为0
    :param if_smooth: 是否平滑矢量
    :return:
    """
    def smoothing(inShp, fname, bdistance=0.001):
        """
        :param inShp: 输入的矢量路径
        :param fname: 输出的矢量路径
        :param bdistance: 缓冲区距离
        :return:
        """
        ogr.UseExceptions()
        in_ds = ogr.Open(inShp)
        in_lyr = in_ds.GetLayer()
        # 创建输出Buffer文件
        driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(fname):
            driver.DeleteDataSource(fname)
        # 新建DataSource, Layer
        out_ds = driver.CreateDataSource(fname)
        out_lyr = out_ds.CreateLayer(fname, in_lyr.GetSpatialRef(), ogr.wkbPolygon)
        def_feature = out_lyr.GetLayerDefn()
        # 遍历原始的Shapefile文件给每个Geometry做Buffer操作
        for feature in in_lyr:
            geometry = feature.GetGeometryRef()
            buffer = geometry.Buffer(bdistance).Buffer(-bdistance)
            out_feature = ogr.Feature(def_feature)
            out_feature.SetGeometry(buffer)
            out_lyr.CreateFeature(out_feature)
            out_feature = None
        out_ds.FlushCache()
        del in_ds, out_ds
    def raster2poly(raster, outshp, geology_dict=dict):
        """栅格转矢量
        Args:
            raster: 栅格文件名
            outshp: 输出矢量文件名
            geology_dict: 地质类型字典, 如{1: "变质岩", 2: "沉积岩", ...}
        """
        inraster = gdal.Open(raster)  # 读取路径中的栅格数据
        inband = inraster.GetRasterBand(1)  # 这个波段就是最后想要转为矢量的波段
        prj = osr.SpatialReference()
        prj.ImportFromWkt(inraster.GetProjection())  # 读取栅格数据的投影信息

        drv = ogr.GetDriverByName("ESRI Shapefile")
        if os.path.exists(outshp):  # 若文件已经存在, 则删除它继续重新做一遍
            drv.DeleteDataSource(outshp)
        Polygon = drv.CreateDataSource(outshp)  # 创建一个目标文件
        Poly_layer = Polygon.CreateLayer(
            raster[:-4], srs=prj, geom_type=ogr.wkbMultiPolygon)

        newField = ogr.FieldDefn('pValue', ogr.OFTReal)
        Poly_layer.CreateField(newField)

        if geology_dict is not None:
            dzField = ogr.FieldDefn('dz', ogr.OFTString)
            dzField.SetWidth(50)  # 设置字段宽度
            Poly_layer.CreateField(dzField)

        gdal.FPolygonize(inband, None, Poly_layer, 0)  # 核心函数, 执行栅格转矢量操作
        if geology_dict is not None:
            for feature in Poly_layer:
                pvalue = feature.GetField('pValue')
                if pvalue in geology_dict:
                    feature.SetField('dz', geology_dict[pvalue])
                    Poly_layer.SetFeature(feature)
        
        Polygon.SyncToDisk()
        Polygon = None
    if os.path.isdir(tif_dir):
        listpic = search_files_in_directory(tif_dir, extension)
    else:
        listpic = [tif_dir]
        tif_dir = os.path.dirname(tif_dir)
    for img in tqdm(listpic):
        tif_img_full_path = img
        base_name = os.path.basename(img)
        shp_full_path = shp_img_path + '/' + base_name[:-4] + '.shp'

        raster2poly(tif_img_full_path, shp_full_path, dict)

        ogr.RegisterAll()  # 注册所有的驱动

        driver = ogr.GetDriverByName('ESRI Shapefile')
        shp_dataset = ogr.Open(shp_full_path, 1)  # 0只读模式, 1读写模式
        if shp_full_path is None:
            print('Failed to open shp_1')

        ly = shp_dataset.GetLayer()

        '''删除矢量化结果中为背景的要素'''
        feature = ly.GetNextFeature()
        while feature is not None:
            gridcode = feature.GetField('pValue')
            if gridcode == delete_value:
                delID = feature.GetFID()
                ly.DeleteFeature(int(delID))
            feature = ly.GetNextFeature()
        ly.ResetReading()  # 重置
        del shp_dataset
        '''平滑矢量'''
        if if_smooth:
            smooth_shp_full_path = shp_img_path + '/' + 'smooth_' + base_name[:-4] + '.shp'
            smoothing(shp_full_path, smooth_shp_full_path, bdistance=0.15)

def random_split_point_shp(input_shp, output_shp1, output_shp2, num_to_select):
    """
    随机分割点Shapefile为两个新文件
    
    参数:
    input_shp: 输入点Shapefile路径
    output_shp1: 输出Shapefile1路径（包含随机选取的要素）
    output_shp2: 输出Shapefile2路径（包含剩余的要素）
    num_to_select: 要随机选取的要素数量, 如果小于1将按照比例选取
    """
    
    # 确保输入文件存在
    if not os.path.exists(input_shp):
        raise FileNotFoundError(f"输入文件不存在: {input_shp}")
    
    # 打开输入数据源
    driver = ogr.GetDriverByName('ESRI Shapefile')
    in_ds = driver.Open(input_shp, 0)
    if in_ds is None:
        raise RuntimeError(f"无法打开输入文件: {input_shp}")
    
    in_layer = in_ds.GetLayer()
    
    # 获取要素总数
    total_features = in_layer.GetFeatureCount()
    if num_to_select > total_features:
        raise ValueError(f"要选择的要素数量({num_to_select})大于总要素数({total_features})")
    if num_to_select < 1:
        num_to_select = int(total_features * num_to_select) # 按比例选取
    # 生成随机索引列表
    indices = list(range(total_features))
    random.shuffle(indices)
    selected_indices = set(indices[:num_to_select])
    
    # 获取输入图层的空间参考和字段定义
    spatial_ref = in_layer.GetSpatialRef()
    layer_defn = in_layer.GetLayerDefn()
    
    # 创建输出数据源1（选中的要素）
    if os.path.exists(output_shp1):
        driver.DeleteDataSource(output_shp1)
    out_ds1 = driver.CreateDataSource(output_shp1)
    out_layer1 = out_ds1.CreateLayer(os.path.basename(output_shp1)[:-4], 
                                    spatial_ref, 
                                    ogr.wkbPoint)
    
    # 复制字段定义
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        out_layer1.CreateField(field_defn)
    
    # 创建输出数据源2（剩余的要素）
    if os.path.exists(output_shp2):
        driver.DeleteDataSource(output_shp2)
    out_ds2 = driver.CreateDataSource(output_shp2)
    out_layer2 = out_ds2.CreateLayer(os.path.basename(output_shp2)[:-4], 
                                    spatial_ref, 
                                    ogr.wkbPoint)
    
    # 复制字段定义
    for i in range(layer_defn.GetFieldCount()):
        field_defn = layer_defn.GetFieldDefn(i)
        out_layer2.CreateField(field_defn)
    
    # 重置图层读取位置
    in_layer.ResetReading()
    
    # 遍历所有要素并根据索引分配到不同的输出文件
    for idx, in_feature in enumerate(in_layer):
        if idx in selected_indices:
            out_layer1.CreateFeature(in_feature.Clone())
        else:
            out_layer2.CreateFeature(in_feature.Clone())
    
    # 清理资源
    in_ds = None
    out_ds1 = None
    out_ds2 = None
    
    print(f"处理完成!")
    print(f"已随机选择 {num_to_select} 个要素保存到: {output_shp1}")
    print(f"剩余 {total_features - num_to_select} 个要素保存到: {output_shp2}")
def batch_random_split_point_shp(input_shp_dir, output_dir, num_to_select):
    """
    批量随机分割点Shapefile为两个新文件
    
    参数:
    input_shp_dir: 输入点Shapefile目录路径
    output_dir: 输出目录路径，包含分割后的Shapefiles
    num_to_select: 要随机选取的要素数量
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isdir(input_shp_dir):
        shp_files = search_files_in_directory(input_shp_dir, extension='.shp')
        output_dir1 = os.path.join(output_dir, "split_part1")
        output_dir2 = os.path.join(output_dir, "split_part2")
        if not os.path.exists(output_dir1):
            os.makedirs(output_dir1)
            os.makedirs(output_dir2)
        if not shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {input_shp_dir}')
        for shp_file in shp_files:
            base_name = os.path.basename(shp_file)[:-4]
            output_shp1 = os.path.join(output_dir1, f"{base_name}_part1.shp")
            output_shp2 = os.path.join(output_dir2, f"{base_name}_part2.shp")
            random_split_point_shp(shp_file, output_shp1, output_shp2, num_to_select)
    elif os.path.isfile(input_shp_dir):
        base_name = os.path.basename(input_shp_dir)[:-4]
        output_shp1 = os.path.join(output_dir, f"{base_name}_part1.shp")
        output_shp2 = os.path.join(output_dir, f"{base_name}_part2.shp")
        random_split_point_shp(input_shp_dir, output_shp1, output_shp2, num_to_select)
    else:
        raise RuntimeError(f'Invalid input_shp_dir: {input_shp_dir}, it should be a directory or a shapefile path')

if __name__ == '__main__':
    # point_value_merge(r'D:\Data\Hgy\龚鑫涛试验数据\program_data\cluster\research1_gmm24_optimization2 - 副本.shp', [13,23])

    clip_by_multishp(r'd:\Data\Hgy\龚鑫涛试验数据\program_data\cluster\test', r'D:\Data\Hgy\龚鑫涛试验数据\Image\research_GF5.dat', 
                r'd:\Data\Hgy\龚鑫涛试验数据\program_data\cluster\research_GF5_samples_2', 17)