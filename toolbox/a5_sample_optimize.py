import sys, os
sys.path.append('.')
try:
    from osgeo import gdal,ogr,osr
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import ShuffleSplit
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from utils import search_files_in_directory, read_csv_to_matrix


def plot_multiline(dataset, curve_types=None, save_path=None, 
                   show_confidence=True):
    """
    优化后的多曲线绘图函数，支持曲线分类显示、置信区间和异常点标记
    
    参数:
        dataset: 二维numpy数组 (n_curves, n_points)
        curve_types: 一维numpy数组 (n_curves,)，值为0或1，标记曲线类型
        title: 图表标题
        save_path: 图片保存路径
        show_confidence: 是否显示置信区间
        show_points: 是否显示每个数据点
        show_outliers: 是否标记异常点
    """
    # 设置颜色（类型0和类型1分别用不同颜色）
    COLOR_TYPE0 = '#1f77b4'  # 蓝色
    COLOR_TYPE1 = '#ff7f0e'  # 橙色
    
    plt.figure(figsize=(10, 6), dpi=125)
    # 设置黑色边框
    with plt.rc_context({'axes.edgecolor': 'black',
                        'axes.linewidth': 1.5}):
        ax = plt.gca()

    # 分离保留和剔除的曲线
    keep_mask = curve_types == 1
    remove_mask = curve_types == -1
    # 计算置信区间（仅基于保留样本）
    if show_confidence and np.any(keep_mask):
        valid_data = dataset[keep_mask]
        mean = np.mean(valid_data, axis=0)
        std = np.std(valid_data, axis=0)
        upper_bound = mean + 1.96 * std
        lower_bound = mean - 1.96 * std
        
        plt.fill_between(range(dataset.shape[1]), lower_bound, upper_bound,
                         color=COLOR_TYPE0, alpha=0.2, 
                         label='95% Confidence (Kept Samples)')
    
    # 检查曲线类型输入
    if curve_types is None:
        curve_types = np.zeros(dataset.shape[0], dtype=int)
    else:
        assert len(curve_types) == dataset.shape[0], "curve_types长度必须与dataset行数一致"
        assert set(np.unique(curve_types)) <= {-1, 1}, "curve_types只能包含0和1"
    # 绘制剔除的曲线
    for i in np.where(remove_mask)[0]:
        plt.plot(dataset[i], color=COLOR_TYPE1, alpha=0.6, linewidth=0.5, zorder=1, linestyle='--', label='Deleted samples')
    # 绘制保留的曲线
    for i in np.where(keep_mask)[0]:
        plt.plot(dataset[i], color=COLOR_TYPE0, alpha=0.8, linewidth=0.5, zorder=2, linestyle='-', label='Saved samples')
    # 坐标轴和网格美化
    ax.grid(True, 
            linestyle='--', 
            linewidth=0.3, 
            alpha=0.5, 
            color='black')  # 黑色虚线网格
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))  
    # 添加图例（只显示每种类型一次）
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))  # 去重
    plt.legend(by_label.values(), by_label.keys(), loc='best')
    
    if save_path:
        count = 1
        base_path = save_path
        while os.path.exists(save_path):
            save_path = base_path[:-4]+str(count)+'.png'
            count+=1
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'png file created:{save_path}')
    plt.close()

def get_idx_from_vector(shp_path, field_name='Emb_Idx'):
    """从shp文件中获取有效像元的索引列表"""
    # 打开Shapefile文件
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shp_path)
    if ds is None:
        raise RuntimeError(f"无法打开Shapefile文件: {shp_path}")
    
    layer = ds.GetLayer()
    layer_defn = layer.GetLayerDefn()  # 获取图层定义
    
    # 检查是否存在Emb_Idx字段
    emb_idx_field = layer_defn.GetFieldIndex(field_name)
    has_emb_idx = emb_idx_field != -1  # 标记是否存在该字段
    
    if not has_emb_idx:
        raise ValueError(f"Shapefile中未找到 {field_name} 字段")
    idx_list = [] # 储存索引值
    valid_features = []
    for feature in layer:
        # 首先检查Emb_Idx字段是否有效
        if has_emb_idx:
            emb_idx = feature.GetField(emb_idx_field)
            if emb_idx is not None:
                idx_list.append(emb_idx)
                valid_features.append(feature.Clone())  # 克隆要素避免引用问题
    ds = None  # 释放数据源
    return valid_features, idx_list

def load_point_coordinates(point_shp, input_img):
    """读取影像范围内有效点的坐标"""
    im_dataset = gdal.Open(input_img)
    if im_dataset is None:
        raise RuntimeError(f"无法打开影像文件: {input_img}")
    im_geotrans = im_dataset.GetGeoTransform()
    im_width = im_dataset.RasterXSize
    im_height = im_dataset.RasterYSize
    coords = []
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_dataset = driver.Open(point_shp, 1)  # 1 表示可写
    if shp_dataset is None:
        raise RuntimeError(f"无法打开矢量文件: {point_shp}")
        
    layer = shp_dataset.GetLayer()

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
            coords.append((x, y))
    shp_dataset = None
    im_dataset = None
    if not coords:
        raise RuntimeError("没有找到影像范围内的有效点")
    return coords

def get_spectral_from_shp(input_img, point_shp):
    '''从shp文件和输入影像中提取光谱数据'''
    coords = load_point_coordinates(point_shp, input_img)
    im_dataset = gdal.Open(input_img)
    if im_dataset is None:
        raise RuntimeError(f"无法打开影像文件: {input_img}")
    im_bands = im_dataset.RasterCount

    out_data = np.zeros((len(coords), im_bands), dtype=np.float32)
    for idx, coord in enumerate(coords):
        x, y = coord
        if im_bands > 1:
            data = im_dataset.ReadAsArray(x, y, 1, 1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) * 1e-4
        else:
            data = im_dataset.GetRasterBand(1).ReadAsArray(x, y, 1, 1)
            if data.dtype == np.int16:
                data = data.astype(np.float32) * 1e-4
        data = data.squeeze()
        out_data[idx] = data
    im_dataset = None

    return out_data

def create_filtered_shapefile_gdal(valid_features, output_shp_path):
    """
    使用GDAL创建新的shapefile，排除指定索引的元素
    
    参数:
        valid_features: 有效的要素列表（ogr.Feature对象列表）
        output_shp_path: 输出的shapefile路径
    """
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_shp_path), exist_ok=True)
    
    # 1. 获取第一个要素的几何类型和字段定义（用于创建新文件）
    if not valid_features:
        raise ValueError("valid_features不能为空")
    
    first_feature = valid_features[0]
    geom_type = first_feature.GetGeometryRef().GetGeometryType()
    src_layer_defn = first_feature.GetDefnRef()
    
    # 2. 创建输出shapefile
    count = 1
    base_path = output_shp_path
    while os.path.exists(output_shp_path):
        output_shp_path = base_path[:-4]+str(count)+'.shp'
        count+=1
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(output_shp_path):
        driver.DeleteDataSource(output_shp_path)
    
    ds_out = driver.CreateDataSource(output_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(first_feature.GetGeometryRef().GetSpatialReference().ExportToWkt())
    
    # 创建图层（使用第一个要素的几何类型和字段定义）
    layer_out = ds_out.CreateLayer(
        os.path.basename(output_shp_path)[:-4],  # 去除.shp后缀
        srs=srs,
        geom_type=geom_type
    )
    # 复制字段定义
    for i in range(src_layer_defn.GetFieldCount()):
        field_defn = src_layer_defn.GetFieldDefn(i)
        layer_out.CreateField(field_defn)
    
    # 3. 过滤并写入要素
    for idx, feature in enumerate(valid_features):
        # 创建新要素
        out_feature = ogr.Feature(layer_out.GetLayerDefn())
        # 复制几何
        geom = feature.GetGeometryRef().Clone()
        out_feature.SetGeometry(geom)
        # 复制属性
        for i in range(src_layer_defn.GetFieldCount()):
            out_feature.SetField(i, feature.GetField(i))
        # 添加到输出图层
        layer_out.CreateFeature(out_feature)
        out_feature = None
    # 4. 释放资源
    ds_out = None
    print(f"shapefile created: {output_shp_path}")

def cal_threshold(arr, th=0.2):
    """
    将数组中前th大的值标记为-1，其余标记为1
    参数:
        arr: 一维numpy数组
        
    返回:
        二值化的一维numpy数组（dtype=int）
    """
    th = (1- th) * 100
    threshold = np.percentile(arr, th)
    binary_arr = np.where(arr > threshold, -1, 1)
    return binary_arr

def process_split(args):
    '''孤立森林评分'''
    X, train_idx, test_idx = args
    model = IsolationForest(contamination=0.1, random_state=42, n_jobs=1, n_estimators=100)
    model.fit(X[train_idx])
    return test_idx, model.decision_function(X) # 异常分数评分，分数越小越异常

def mccv_lf(X, test_size=0.2, n_splits=1000, n_jobs=-1):
    '''多进程蒙特卡洛'''
    anomaly_scores = np.zeros(X.shape[0]) # 记录异常分数
    nums_list = np.zeros(X.shape[0]) + 1e-4 # 记录成为测试集的次数
    rs = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    
    # 准备任务参数
    tasks = [(X, train_idx, test_idx) for train_idx, test_idx in rs.split(X)]
    
    # 使用多进程并行处理
    with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
        results = list(tqdm(
            executor.map(process_split, tasks),
            total=n_splits,
            desc="Processing splits"
        ))
    # 合并结果
    for test_idx, scores in results:
        anomaly_scores += scores
        # nums_list[test_idx] += 1
    anomaly_scores /= n_splits
    return anomaly_scores

def lf(data, contamination=0.2, useless=None):
    """
    使用Isolation Forest剔除异常样本
    
    参数：
        data: 输入数据矩阵 (n_samples, n_features)
        contamination: 异常样本比例（默认0.2即20%）
        useless: 无用，不用管
    返回：
        cleaned_data: 清洗后的数据（去除异常点）
        anomaly_indices: 被判定为异常的样本索引
    """
    # 训练孤立森林模型
    clf = IsolationForest(contamination=contamination, 
                         random_state=42,
                         n_estimators=100)
    clf.fit(data)
    preds = clf.predict(data)
    return preds

input_img = r"C:\Users\85002\OneDrive - cugb.edu.cn\项目数据\张川铀资源\ZY_result\Image\research_area1.dat"
input_shp_dir = r'c:\Users\85002\Desktop\TempDIR\test3\RESEARCH_GF5_1'

embeddings = r'C:\Users\85002\Desktop\TempDIR\embeddings2.csv'
func = "LF" # 两个选项 Mccv_LF LF
data_form = 'Spectral' # 两个选项 Spectral Embedding
ratio = 0.2
n_splits = 1000
if __name__ == '__main__':
    algorithm = mccv_lf if func == "Mccv_LF" else lf
    if os.path.isdir(input_shp_dir):
        point_shp_files = search_files_in_directory(input_shp_dir, extension='.shp')
        if not point_shp_files:
            raise RuntimeError(f'Not shapefiles found in directory: {input_shp_dir}')
        base_dir = os.path.dirname(input_shp_dir)
        dir_name = os.path.basename(input_shp_dir)
        output_dir_base = os.path.join(base_dir, f"{dir_name}_optimize")
        output_dir = output_dir_base
        count = 1
        while os.path.exists(output_dir):
            output_dir = f"{output_dir_base}{count}"
            count += 1
        os.makedirs(output_dir)
        for idx, input_shp in enumerate(point_shp_files):
                output_name = os.path.basename(input_shp)[:-4]
                output_path = os.path.join(output_dir, output_name+"_optimize.shp") # shp文件输出绝对地址
                out_png_path = os.path.join(output_dir, output_name+"_optimize.png")

                valid_features, idx_list = get_idx_from_vector(input_shp) # 获取点元素
                if data_form == "Spectral":
                    data = get_spectral_from_shp(input_img, input_shp) # 获取所有点的光谱数据
                else:
                    if embeddings is not None:
                        data = read_csv_to_matrix(embeddings)
                        mask = idx_list
                        idx_list = [i-1 for i in mask if i-1 >= 0] # 获取有embeding的点索引
                        data = data[idx_list] # 获取ebedding数据
                    else: raise ValueError('Must choose the embedding file!')
                scores = algorithm(data, ratio, n_splits) # 计算异常评分
                valid_mask = cal_threshold(-scores, th=ratio) if func == "Mccv_LF" else scores # 保留样本为1，剔除样本为-1
                valid_features = [valid_features[i] for i in range(len(valid_features)) if valid_mask[i]==1]
                create_filtered_shapefile_gdal(valid_features, output_shp_path=output_path)
                plot_multiline(data, valid_mask, save_path=out_png_path)
    elif os.path.isfile(input_shp_dir):
        print('Note:the input is a single shape file')
        output_dir = os.path.dirname(input_shp_dir)
        output_name = os.path.basename(input_shp_dir)[:-4]
        output_path = os.path.join(output_dir, output_name+"_optimized.shp") # shp文件输出绝对地址
        out_png_path = os.path.join(output_dir, output_name+"_optimized.png")

        valid_features, idx_list = get_idx_from_vector(input_shp_dir) # 获取点元素
        if data_form == "Spectral":
            data = get_spectral_from_shp(input_img, input_shp_dir) # 获取所有点的光谱数据
        else:
            if embeddings is not None:
                data = read_csv_to_matrix(embeddings)
                mask = idx_list
                idx_list = [i-1 for i in mask if i-1 >= 0] # 获取有embeding的点索引
                data = data[idx_list] # 获取ebedding数据
            else: raise ValueError('Must choose the embedding file!')
        scores = algorithm(data, ratio, n_splits) # 计算异常评分
        valid_mask = cal_threshold(-scores, th=ratio) if func == "Mccv_LF" else scores # 保留样本为1，剔除样本为-1
        valid_features = [valid_features[i] for i in range(len(valid_features)) if valid_mask[i]==1]
        create_filtered_shapefile_gdal(valid_features, output_shp_path=output_path) # 创建新shp文件
        plot_multiline(data, valid_mask, save_path=out_png_path) # 绘制曲线图
    else:
        raise RuntimeError(f'Invalid point_shp_dir: {input_shp_dir}, it should be a directory or a shapefile path')