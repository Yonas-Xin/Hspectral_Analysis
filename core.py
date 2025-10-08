try:
    from osgeo import gdal
except ImportError:
    print('gdal is not used')
import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from gdal_utils import *
from utils import save_matrix_to_csv
import spectral as spy
from skimage.segmentation import slic
from utils import block_generator
from algorithms import *
from skimage.feature import graycomatrix, graycoprops

gdal.UseExceptions()
class Hyperspectral_Image:
    '''如果数据是int16类型，自动缩放为0-1的光谱反射率范围'''
    def __init__(self):
        self.dataset, self.rows, self.cols, self.bands = None, None, None, None
        self.no_data = None
        self.sampling_position = None # 二维矩阵，标记了样本的取样点和类别信息
        self.backward_mask = None # [rows, cols] 背景掩膜
        self.ori_img = None # [rows, cols, 3] 拉伸影像
        self.enhance_data = None # [rows, cols, bands] 增强原始数据
        self.enhance_img = None #[rows, cols, 3] 增强-拉伸影像
        self.geotransform = None
        self.projection = None

        self.slic_label = None # [rows， cols] 
        self.slic_img = None # [rows, cols, 3]
    def __del__(self):
        self.dataset = None # 释放内存

    def init(self, filepath, init_fig=True, rgb=(1,2,3)):
        try:
            dataset = gdal.Open(filepath)
            bands = dataset.RasterCount
            rows, cols = dataset.RasterYSize, dataset.RasterXSize
            self.dataset, self.rows, self.cols, self.bands = dataset, rows, cols, bands
            self.geotransform = dataset.GetGeoTransform()
            self.projection = dataset.GetProjection()
            if init_fig: # 根据需要加载影像数据
                self.init_fig_data(rgb=rgb)
            return 0 # 代表数据导入成功
        except (AttributeError,RuntimeError):
            return 1

    def create_vector(self,mask,out_file): # mask 转单矢量点
        mask_to_vector_gdal(mask, self.geotransform, self.projection,
                                   output_shapefile=out_file)
    def create_mutivector(self, mask_matrix, out_dir, out_dir_name): # mask 转多矢量点
        mask_to_multivector(mask_matrix, self.geotransform, self.projection,
                            output_dir=out_dir, output_dir_name=out_dir_name)
        
    def create_mask(self, input_file): # 矢量点转mask，点的数值由“class”字段确定
        return vector_to_mask(input_file, self.geotransform, self.rows, self.cols)
    
    def create_mask_from_mutivector(self, inputdir): # 多矢量点转mask
        return mutivetor_to_mask(inputdir, self.geotransform, self.rows, self.cols)
    
    def save_tif(self, filename, img_data, nodata = None, mask=None):
        '''将(rows, cols,  bands)或(rows, cols)的数据存为tif格式, tif具有与img相同的投影信息'''
        nodata = self.no_data if nodata is None else nodata
        if len(img_data.shape) == 3:
            write_data_to_tif(filename, img_data.transpose(2,0,1), self.geotransform, self.projection,
                          nodata_value=nodata, mask=mask)
        elif len(img_data.shape) == 2:
            write_data_to_tif(filename, img_data, self.geotransform, self.projection,
                nodata_value=nodata, mask=mask)
        else:
            raise ValueError("The input dims must be 2 or 3")
        return True
    
    def face_vector_to_mask(self, shp_file): # 矢量面转mask
        return face_vector_to_mask(shp_file, self.geotransform, self.projection, self.rows, self.cols)

    def init_fig_data(self, rgb = (1,2,3)): # 计算背景掩膜，生成拉伸图像
        band = self.dataset.GetRasterBand(1)
        self.no_data = band.GetNoDataValue()
        self.backward_mask = self.ignore_backward()  # 初始化有效像元位置
        r,g,b = rgb
        self.compose_rgb(r=r, g=g, b=b)

    def update(self,r,g,b,show_enhance_img=False): # 根据所选择rgb组合更新拉伸图像
        if show_enhance_img:
            self.compose_enhance(r,g,b)
        else:
            self.compose_rgb(r,g,b)

    def compose_rgb(self, r, g, b, stretch=True): # 合成彩色图像
        try:
            r_band = self.get_band_data(r)
            g_band = self.get_band_data(g)
            b_band = self.get_band_data(b)
        except:
            print("波段序号最小为1！")
        try:# 拉伸出错可能是mask全为False，忽略
            if stretch:
                r_band = linear_2_percent_stretch(r_band, self.backward_mask)
                g_band = linear_2_percent_stretch(g_band, self.backward_mask)
                b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        except ValueError as e:
            print(f'Error in linear stretch: {e}')
            r_band = r_band[self.backward_mask]
            g_band = g_band[self.backward_mask]
            b_band = b_band[self.backward_mask]
        rgb = np.dstack([r_band, g_band, b_band]).squeeze().astype(np.float32)
        self.ori_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.ori_img[self.backward_mask] = rgb
        return self.ori_img

    def compose_enhance(self, r, g, b, stretch=True): # 合成增强彩色图像
        '''这里为了和tif波段组合统一，读取enhance_data波段数据，波段减一'''
        r_band = self.enhance_data[:, :, r-1]
        g_band = self.enhance_data[:, :, g-1]
        b_band = self.enhance_data[:, :, b-1]
        try:
            if stretch:
                r_band = linear_2_percent_stretch(r_band, self.backward_mask)
                g_band = linear_2_percent_stretch(g_band, self.backward_mask)
                b_band = linear_2_percent_stretch(b_band, self.backward_mask)
        except ValueError as e:
            print(f'Error in linear stretch: {e}')
            r_band = r_band[self.backward_mask]
            g_band = g_band[self.backward_mask]
            b_band = b_band[self.backward_mask]
        rgb = np.dstack([b_band, g_band, r_band]).squeeze().astype(np.float32)
        self.enhance_img = np.zeros((self.rows, self.cols, 3)) + 1
        self.enhance_img[self.backward_mask] = rgb

    def get_band_data(self, band_idx):
        """获取指定波段的数据
        :return (rows, cols)"""
        if self.dataset is None:
            return None
        band = self.dataset.GetRasterBand(band_idx)
        band_data = band.ReadAsArray()
        if band_data.dtype == np.int16:  # 如果是int16类型数据，进行缩放
            band_data = band_data.astype(np.float32) / 10000
        return band_data

    def get_dataset(self, scale=1e-4):
        '''return: (bands, rows, cols)的numpy数组，数据类型为float32'''
        dataset = self.dataset.ReadAsArray()
        if dataset.dtype == np.int16 and scale != 1: # 如果是int16类型数据并且scale不为1，进行缩放
            dataset = dataset.astype(np.float32) * scale
        return dataset

    def ignore_backward(self):
        '''分块计算背景掩膜值，默认分块大小为512'''
        no_data = self.no_data if self.no_data is not None else nodata_value # 如果原始数据没有nodata值，默认为0
        block_size = 512 if self.cols> (2 * 512) and self.rows > (2 * 512) else min(self.rows, self.cols)
        mask = np.empty((self.rows, self.cols), dtype=bool)
        for i in range(0, self.rows, block_size):
            for j in range(0, self.cols, block_size):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(block_size, self.rows - i)
                actual_cols = min(block_size, self.cols - j)
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                block_mask = np.all(block_data == no_data, axis=0)
                mask[i:i + actual_rows, j:j + actual_cols] = ~block_mask
        return mask

    def image_enhance(self, f='PCA', n_components=10, row_slice=None, col_slice=None, band_slice=None):
        # 影像增强
        no_data = self.no_data if self.no_data is not None else nodata_value # 如果原始数据没有nodata值，默认为0
        ori_dataset = self.get_dataset().transpose(1, 2, 0)
        dataset = ori_dataset[self.backward_mask]
        if f == 'PCA':
            dataset = pca(dataset, n_components=n_components)
        elif f == 'MNF':
            if row_slice is None and col_slice is None and band_slice is None:
                mask = self.backward_mask.astype(np.int16)
            else: mask = None
            row_slice, col_slice, band_slice = to_slice(row_slice), to_slice(col_slice), to_slice(band_slice)
            noise_stats = noise_estimation(ori_dataset[row_slice, col_slice, band_slice], mask=mask)
            dataset = mnf_standard(dataset, noise_stats, n_components)
        self.enhance_data = np.full((self.rows, self.cols, n_components), no_data, dtype=np.float32)
        self.enhance_data[self.backward_mask] = dataset
        self.compose_enhance(1,2,3)
        return self.enhance_img
    
    def slic(self, n_segments, compactness=10, n_components=10):
        '''超像素分割'''
        self.image_enhance(f='PCA', n_components=n_components) # 使用PCA增强
        slic_label = superpixel_segmentation(self.enhance_img, n_segments=n_segments, compactness=compactness, mask=self.backward_mask)
        print(f'The number of super pixels:{np.max(slic_label)}')

        # 生成超像素图像
        slic_img = np.zeros_like(self.enhance_img)
        for seg_val in np.unique(slic_label):  # 遍历所有超像素区域
            mask = slic_label == seg_val  # 选出当前超像素区域
            mean_color = self.enhance_img[mask].mean(axis=0)  # 计算均值颜色
            slic_img[mask] = mean_color  # 赋值给对应的区域
        return slic_label, slic_img

    def ppi(self, niters=2000, threshold=0, centered=False):
        dataset = self.get_dataset()
        dataset = dataset[self.backward_mask]
        dataset = mnf_standard(dataset, self.rows, self.cols, n_components=24)
        dataset = dataset.reshape(self.rows, self.cols, -1)
        dataset = ppi_manual(dataset, niters=niters, threshold=threshold, centered=centered)
        dataset = dataset.reshape(self.rows, self.cols)
        return dataset

    def superpixel_sampling(self, slic_label, enhanced_data,  max_samples=20, niters=2000, threshold=0, centered=False):
        '''超像素分割-ppi采样
        slic_label: 超像素标签矩阵
        enhanced_data: 增强后的数据矩阵
        max_samples: 每个超像素区域的最大采样数量'''
        labels = np.unique(slic_label) # 超像素的标签
        labels = labels[labels > 0]  # 排除背景标签
        out_labels = np.zeros_like(slic_label) # PPI提取结果，最后样本提取结果

        areas = [] # 面积参数
        var_sums = [] # 光谱反差参数
        for label in labels:
            mask = (slic_label == label)
            area = np.sum(mask) # 当前超像素区域的像素数量
            areas.append(area)
            datasets = enhanced_data[mask]

            if datasets.size == 0:
                var_sum = 0.0
            else:
                var_per_band = np.var(datasets, axis=0)  # 波段方差
                var_sum = np.sum(var_per_band)  # 光谱方差总和
            var_sums.append(var_sum)
        areas = np.array(areas, dtype=float)
        var_sums = np.array(var_sums, dtype=float)

        # Min-Max 归一化函数（避免除以零）
        eps = 1e-12
        def min_max_norm(x):
            if np.all(x == x[0]):
                return np.zeros_like(x)
            return (x - x.min()) / (x.max() - x.min() + eps)
        
        norm_var = min_max_norm(var_sums)
        norm_area = min_max_norm(areas)
        area_inv = 1.0 - norm_area  # 面积倒数特征

        num_features = 2  # 包含 area_inv
        scores = (norm_var + area_inv) / num_features # 0~1

        # 根据得分确定采样数量 (scores * 20, 限制 [1,20])
        sample_counts = np.round(scores * max_samples).astype(int)
        sample_counts[sample_counts < 1] = 1
        sample_counts[sample_counts > max_samples] = max_samples

        for idx, label in enumerate(labels):
            mask = (slic_label==label)
            datasets = enhanced_data[mask]
            ppi_label = ppi_manual(datasets, niters=niters, threshold=threshold, centered=centered)

            top_n = sample_counts[idx]
            top_indices = np.argsort(ppi_label)[-top_n:]
            ppi_label_selected = np.zeros_like(ppi_label)
            ppi_label_selected[top_indices] = ppi_label[top_indices]  # 只保留top_n个端元
            out_labels[mask] = ppi_label_selected
        out_labels = (out_labels > 0).astype(np.int8)
        print(f'The number of samples: {np.sum(out_labels)}')
        return out_labels

    # def old_superpixel_sampling(self, n_segments=1024, compactness=10, niters=2000, threshold=0, centered=False,
    #                         samples=8000, embedding_nums=10, f='MNF', row_slice=None, col_slice=None, band_slice=None):
    #     '''超像素分割-ppi采样'''
    #     if self.slic_label is None:
    #         # 如果没有进行超像素分割，则进行分割,同时这里面生成了pca增强影像
    #         self.slic(n_segments=n_segments, compactness=compactness, n_components=embedding_nums, show_img=True)
    #     labels = np.unique(self.slic_label)
    #     out_labels = np.zeros_like(self.slic_label)
    #     if f=='MNF': # 如果是MNF增强，重新计算增强数据，覆盖原来的增强数据
    #         self.image_enhance(f='MNF', n_components=embedding_nums, row_slice=None, col_slice=None, band_slice=None)
    #     mnf_data = self.enhance_data
    #     for label in labels:
    #         if label == 0:  # 跳过背景标签
    #             continue
    #         mask = self.slic_label==label
    #         datasets = mnf_data[mask]
    #         ppi_label = ppi_manual(datasets, niters=niters, threshold=threshold, centered=centered)
    #         out_labels[mask] = ppi_label
    #     valid_mask = out_labels > 0
    #     list = out_labels[valid_mask].tolist()
    #     list.sort(reverse=True) # 从大到小排序
    #     try:
    #         th = list[samples]
    #     except:
    #         th = list[-1]
    #     print(f'the real seg threshold:{th}')
    #     return (out_labels>=th).astype(np.int8)

    def block_generator(self, block_size=512):
        if isinstance(block_size, tuple):
            rows, cols = block_size
        elif isinstance(block_size, int):
            rows = cols = block_data
        else: raise ValueError('the input must be tuple or int')
        for i in range(0, self.rows, rows):
            for j in range(0, self.cols, cols):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(rows, self.rows - i)
                actual_cols = min(cols, self.cols - j)
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=j, yoff=i, xsize=actual_cols, ysize=actual_rows)
                if block_data.dtype == np.int16:
                    block_data = block_data.astype(np.float32) * 1e-4
                yield block_data

    # def block_mnf_ppi(self, block_size=512):
    #     '''分块mnf降维，ppi端元提取'''
    #     out_labels = np.zeros((self.rows, self.cols))
    #     for block, position_mask in zip(self.block_generator(block_size), block_generator(out_labels, block_size)):
    #         bands, rows, cols = block.shape
    #         block = block.transpose(1,2,0).reshape(-1, bands)
    #         mnf_data = mnf_standard(block, rows, cols, n_components=10).reshape(rows,cols,-1)
    #         ppi_label = ppi_manual(mnf_data, niters=2000, threshold=0, centered=False)
    #         out_labels[position_mask==1] = ppi_label
    #     return out_labels
    
    def generate_sampling_mask(self, sample_fraction=0.001):
        """经过该函数进行随机取样，取样位置的标签为1"""
        rows, cols, bands = self.rows, self.cols, self.bands
        mask = self.backward_mask if self.backward_mask is not None else self.ignore_backward()

        total_pixels = rows * cols
        num_samples = int(total_pixels * sample_fraction)
        all_indices = np.arange(total_pixels)
        sampled_indices = np.random.choice(all_indices, size=num_samples, replace=False)
        sampled_rows = sampled_indices // cols
        sampled_cols = sampled_indices % cols
        self.sampling_position = np.zeros((rows, cols), dtype=np.uint8)
        self.sampling_position[sampled_rows, sampled_cols] = 1
        self.sampling_position[~mask] = 0
        return self.sampling_position

    def crop_image_by_mask_block(self, filepath, image_block=256, patch_size=30, position_mask=None, name="Image_"):
        '''分块裁剪样本，适合无法一次加载到内存的大影像，生成一个txt文件（数据排序按照图像块中点行列顺序排序）
        依据的mask矩阵为sampling_position，后续考虑更换。生成的影像自动转化为float32类型，缩放为0-1范围
        return: None 生成txt文本，生成采样位置矩阵csv文件，生成分块影像数据'''
        if position_mask is None:
            position_mask = self.sampling_position
        if position_mask is None or np.max(position_mask)<=0:
            raise ValueError('sampling_position have to be valid!')
        save_matrix_to_csv(position_mask, os.path.join(filepath, '.sampling_position.csv')) # 保存采样位置矩阵
        geotransform = self.geotransform
        projection = self.projection
        left_top = int(patch_size / 2 - 1) if patch_size % 2 == 0 else int(patch_size // 2)
        right_bottom = int(patch_size / 2) if patch_size % 2 == 0 else int(patch_size // 2)
        num = 1
        pathlist = []
        add_labels = False
        actual_indices = []
        for i in range(0, self.rows, image_block):
            for j in range(0, self.cols, image_block):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(image_block+patch_size-1, self.rows - i)#实际高
                actual_cols = min(image_block+patch_size-1, self.cols - j)#实际宽
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
                if actual_cols == (self.cols - j): # 如果实际宽度已经接近了最右边界
                    pad = actual_cols - image_block
                    right_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_cols += left_top
                else:
                    right_pad = 0
                if actual_rows == (self.rows - i):
                    pad = actual_rows - image_block
                    bottom_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_rows += left_top
                else:
                    bottom_pad = 0


                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=actual_cols, ysize=actual_rows)
                if block_data.dtype == np.int16:
                    block_data = block_data.astype(np.float32) * 1e-4
                block_data = np.pad(block_data,[(0, 0), (top_pad, bottom_pad), (left_pad, right_pad)], 'constant')

                row_block = min(image_block, self.rows - i) # 记录真实窗口大小
                col_block = min(image_block, self.cols - j)
                block_sampling_mask = position_mask[i:i + row_block, j:j + col_block]
                oringinx = geotransform[0]+j*geotransform[1]
                oringiny = geotransform[3]+i*geotransform[5]
                if np.all(block_sampling_mask==0):
                    continue
                pbar = tqdm(total=int(np.sum(block_sampling_mask > 0))) # 进度条
                for row in range(row_block):
                    for col in range(col_block):
                        if block_sampling_mask[row, col] > 0:  # 中点位置不是背景和噪声
                            indice = (i+row, j+col)
                            actual_indices.append(indice) # 记录真实indice
                            oringinX = (col-left_top)*geotransform[1]+oringinx
                            oringinY = (row-left_top)*geotransform[5]+oringiny
                            new_geotransform = (oringinX, geotransform[1], geotransform[2], oringinY, geotransform[4], geotransform[5])
                            path = os.path.join(filepath, name + f'{patch_size}_{patch_size}_{num}.tif')
                            block = block_data[:, row:row + patch_size, col:col+patch_size]
                            write_data_to_tif(path, block, geotransform=new_geotransform, projection=projection)
                            if add_labels:
                                pathlist.append(path + f' {block_sampling_mask[row, col] - 1}')
                            else:
                                pathlist.append(path)
                            num+=1
                            pbar.update(1)
        paired = list(zip(actual_indices, pathlist)) # （row，cols）与地址配对
        paired_sorted = sorted(paired, key=lambda x: (x[0][0], x[0][1])) # 按照row，col排序
        sorted_paths = [x[1] for x in paired_sorted]
        dataset_path = os.path.join(filepath, '.datasets.txt')
        write_list_to_txt(sorted_paths, dataset_path) # 打印txt文件
        print('Complete!')

    def block_images(self, image_block=256, block_size=30): # 该迭代器用于预测大影像
        """迭代器，返回分块数据和块的左上角坐标"""
        left_top = int(block_size / 2 - 1) if block_size % 2 == 0 else int(block_size // 2)
        right_bottom = int(block_size / 2) if block_size % 2 == 0 else int(block_size // 2)
        for i in range(0, self.rows, image_block):
            for j in range(0, self.cols, image_block):
                # 计算当前块的实际高度和宽度（避免越界）
                actual_rows = min(image_block + block_size - 1, self.rows - i)  # 实际高
                actual_cols = min(image_block + block_size - 1, self.cols - j)  # 实际宽
                xoff = 0 if (j - left_top) < 0 else j - left_top
                left_pad = left_top if (j - left_top) < 0 else 0
                yoff = 0 if (i - left_top) < 0 else i - left_top
                top_pad = left_top if (i - left_top) < 0 else 0
                if (j - left_top) < 0: actual_cols -= left_top
                if (i - left_top) < 0: actual_rows -= left_top
                # 计算边缘pad
                if actual_cols == (self.cols - j): # 如果实际宽度已经接近了最右边界
                    pad = actual_cols - image_block
                    right_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_cols += left_top
                else:
                    right_pad = 0
                if actual_rows == (self.rows - i):
                    pad = actual_rows - image_block
                    bottom_pad = right_bottom - pad if pad >=0 else right_bottom
                    actual_rows += left_top
                else:
                    bottom_pad = 0
                # 读取当前块的所有波段数据（形状: [bands, actual_rows, actual_cols]）
                block_data = self.dataset.ReadAsArray(xoff=xoff, yoff=yoff, xsize=actual_cols, ysize=actual_rows)
                if block_data.dtype == np.int16:
                    block_data = block_data.astype(np.float32) * 1e-4
                block_data = np.pad(block_data, [(0, 0), (top_pad, bottom_pad), (left_pad, right_pad)], 'constant')
                # 经过上面的计算位于左上区域和中间区域的块大小一律为（image_block + block_size - 1，image_block + block_size - 1）
                # 比如如果参数是64， 17， 那么裁剪的块大小为（80, 80）
                row_block = min(image_block, self.rows - i) # 记录真实窗口大小
                col_block = min(image_block, self.cols - j)
                block_sampling_mask = self.backward_mask[i:i + row_block, j:j + col_block]
                yield block_data, block_sampling_mask, i, j
    
    def read_dataset_from_mask(self, mask): # 根据mask读取数据，形成数据集
        """大影像读取数据集"""
        mask = mask.astype(np.int16)
        index_list = []
        label_list = []
        for y in range(self.rows):
            for x in range(self.cols):
                mask_value = mask[y, x]
                if mask_value != 0:  # 0表示背景，跳过
                    index_list.append((x, y))
                    label_list.append(mask_value - 1)  # 标签从0开始
        label = np.array(label_list, dtype=np.int16)
        num_samples = len(label_list)

        data = np.zeros((num_samples, self.bands), dtype=np.float32)
        for i, (x, y) in enumerate(index_list):
            pixel_data = self.dataset.ReadAsArray(xoff=x, yoff=y, xsize=1, ysize=1).flatten()
            if pixel_data.dtype == np.int16:
                pixel_data = pixel_data.astype(np.float32) * 1e-4
            data[i, :] = pixel_data
        return data, label
    
    def Mianvector2raster(self, input_shp, out_tif, nodata=0, fill_value=255):
        '''将矢量面转为栅格，矢量内区域设为fill_value，外部区域设为0，处理结果与原始数据大小一致'''
        mask = Mianvector2mask(self.geotransform, self.projection, width=self.cols, height=self.rows, vector_path=input_shp, fill_value=fill_value)
        mask = mask.astype(np.uint8)
        self.save_tif(out_tif, mask, nodata=nodata)
    
    def Mianvector_clip_tif(self, input_shp, out_tif, nodata=0, fill_value=1):
        '''根据面shp文件对影像进行裁剪，矢量内区域保留，外部区域设为nodata，处理结果与原始数据大小一致'''
        mask = Mianvector2mask(self.geotransform, self.projection, width=self.cols, height=self.rows, vector_path=input_shp, fill_value=fill_value)
        mask = mask.astype(bool)
        self.save_tif(out_tif, self.get_dataset(scale=1), nodata=nodata, mask=mask) # 保存裁剪影像，原始数据不缩放
        
def to_slice(s=None):
    """s:(int, int) or int or None"""
    if s is None:
        return slice(None)
    else:return slice(*s) if isinstance(s, tuple) else s
    
def linear_2_percent_stretch(band_data, mask=None):
    '''
    线性拉伸
    :param band_data: 单波段数据[rows, cols]
    :param mask: [rows, cols], bool类型，True表示有效像元
    :return: stretched_band[valid_pixels,]
    '''
    if mask is not None and np.sum(mask) == 0:
        return np.zeros_like(band_data)
    band_data = band_data[mask] if mask is not None else band_data.reshape(band_data.shape[0]*band_data.shape[1])
    # 计算2%和98%分位数
    lower_percentile = np.percentile(band_data, 2)
    upper_percentile = np.percentile(band_data, 98)
    if lower_percentile == upper_percentile:
        return np.zeros_like(band_data)
    # 拉伸公式：将数值缩放到 0-1 范围内
    stretched_band = np.clip((band_data - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    return stretched_band

def linear_percent_stretch(band_data, mask=None):
    if mask is not None and np.sum(mask) == 0:
        return np.zeros_like(band_data)
    band_data = band_data[mask] if mask is not None else band_data.reshape(band_data.shape[0]*band_data.shape[1])
    # 计算2%和98%分位数
    lower_percentile = np.min(band_data)
    upper_percentile = np.max(band_data)
    if lower_percentile == upper_percentile:
        return np.zeros_like(band_data)
    # 拉伸公式：将数值缩放到 0-1 范围内
    stretched_band = np.clip((band_data - lower_percentile) / (upper_percentile - lower_percentile), 0, 1)
    return stretched_band

def pca(data, n_components=10):
    ''':param data: [rows*cols，bands]'''
    # 计算协方差矩阵
    covariance_matrix = np.cov(data, rowvar=False)
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    # 按特征值降序排序特征向量
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors_sorted = eigenvectors[:, sorted_idx]
    eigenvectors_selected = -eigenvectors_sorted[:, :n_components] #这里取了负值，实际上正值负值不会影响数据分布，只会影响影像呈现

    data = np.dot(data, eigenvectors_selected)
    return data

def estimate_noise_highpass_non_square(data, sigma=1):
    """
    高通滤波法估计噪声，适用于非方形数据。

    参数:
        data: numpy.ndarray, 输入数据，形状为 (num_samples, bands)
        sigma: float, 高斯滤波器标准差

    返回:
        noise: numpy.ndarray, 噪声数据，形状与输入数据相同
    """
    smoothed = gaussian_filter1d(data, sigma=sigma, axis=1).astype(np.float32)
    noise = data - smoothed
    return noise

def mnf_standard(dataset, noise_stats, n_components=10):
    """
    dataset: [rows, cols, bands]
    noise_stats: 噪声统计量
    """
    data_stats = signal_estimation(dataset)
    mnf_result = spy.mnf(data_stats, noise_stats)
    return mnf_result.reduce(dataset, num=n_components)

def Scaler(data, std = False):
    '''
    对数据进行中心化或者标准化
    std: False-中心化 True-标准化
    '''
    scaler = StandardScaler(with_mean=True, with_std=std)
    return scaler.fit_transform(data)

def superpixel_segmentation(data_pca, n_segments=1024, compactness=10, mask=None):
    segments = slic(
        data_pca,
        n_segments=n_segments,
        compactness=compactness,
        start_label=1,
        mask=mask,
    )
    return segments

def ppi(data, niters=2000, threshold=0, centered=False):
    '''data:[rows, cols, bands]'''
    return spy.ppi(data, niters=niters, threshold=threshold, centered=centered)

def ppi_manual(X, niters, threshold=0, centered=False):
    '''根据spectral库ppi算法改动
    X：[rows, cols, bands] or [nums, bands]
    :return counts[nums, ] 数值表示被选为纯净像元的次数
    threshold: 越大提取的纯净像元越多'''
    if not centered:
        stats = spy.calc_stats(X)
        X = X - stats.mean
    nbands = X.shape[-1]
    if X.ndim == 3:
        X = X.reshape(-1, nbands)
    counts = np.zeros(X.shape[0], dtype=np.uint32)
    for i in tqdm(np.arange(niters), total=niters):
        r = np.random.rand(nbands) - 0.5
        r /= np.sqrt(np.sum(r * r))
        s = X.dot(r)
        imin = np.argmin(s)
        imax = np.argmax(s)

        if threshold == 0:
            # Only the two extreme pixels are incremented
            counts[imin] += 1
            counts[imax] += 1
        else:
            # All pixels within threshold distance from the two extremes
            counts[s >= (s[imax] - threshold)] += 1
            counts[s <= (s[imin] + threshold)] += 1
    return counts

if __name__ == '__main__':
    x = np.random.rand(100, 10)
    slice = slice(None)
    y = x[slice]
    print(y.shape)