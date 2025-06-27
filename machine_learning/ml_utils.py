import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import numpy as np
try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    print('gdal is not used')
from tqdm import tqdm
from utils import rewrite_paths_info
def read_tif_with_gdal(tif_path):
    '''读取栅格原始数据
    返回dataset[bands,H,W]'''
    dataset = gdal.Open(tif_path)
    dataset = dataset.ReadAsArray()
    if dataset.dtype == np.int16:
        dataset = dataset.astype(np.float32) * 1e-4
    return dataset

def load_dataset_from_txt(txt_path):
    data = []
    labels = []
    dataset_list = rewrite_paths_info(txt_path)
    for line in tqdm(dataset_list, total=len(dataset_list), desc="Loading data"):
        path, label = line.strip().split()
        spectrum = read_tif_with_gdal(path).squeeze()  # (bands,)
        data.append(spectrum)
        labels.append(int(label))
    data = np.stack(data)     # shape: (N, bands)
    labels = np.array(labels) # shape: (N,)
    return data, labels