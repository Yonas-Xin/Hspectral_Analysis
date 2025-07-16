"""从spectral库中重新封装算法，对部分算法做了调整"""
import spectral as spy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
def noise_from_diffs(X, mask=None, direction='lowerright'):

    if direction.lower() not in ['lowerright', 'lowerleft', 'right', 'lower']:
        raise ValueError('Invalid `direction` value.')
    if mask is not None and mask.dtype != np.bool:
        mask = mask.astype(np.bool)
    if direction == 'lowerright':
        deltas = X[:-1, :-1, :] - X[1:, 1:, :]
        if mask is not None:
            mask = mask[:-1, :-1] & mask[1:, 1:]
    elif direction == 'lowerleft':
        deltas = X[:-1, 1:, :] - X[1:, :-1, :]
        if mask is not None:
            mask = mask[:-1, 1:] & mask[1:, :-1]
    elif direction == 'right':
        deltas = X[:, :-1, :] - X[:, 1:, :]
        if mask is not None:
            mask = mask[:, :-1] & mask[:, 1:]
    else:
        deltas = X[:-1, :, :] - X[1:, :, :]
        if mask is not None:
            mask = mask[:-1, :] & mask[1:, :]

    stats = spy.calc_stats(deltas, mask=mask) # 引入mask，忽略背景值统计值的计算
    stats.cov /= 2.0
    return stats

def noise_estimation(data, mask=None):
    # data[rows, cols, bands]
    # 差分法估计噪声
    # return 噪声统计量
    return noise_from_diffs(data, mask)

def signal_estimation(data):
    # data[rows, cols, bands] or [nums, bands]
    # 计算全局统计量
    # return 信号统计量
    return spy.calc_stats(data)


def spectral_complexity_pca(data):
    """
    输入数据，计算复杂度指标（基于PCA解释方差比例）。
    参数：
    data: np.ndarray，形状为 (样本数, 波段数)
    返回：
    complexity: float，复杂度指标，定义为 1 - 第一主成分解释方差比例
                越接近0表示数据差异小，越接近1表示复杂度高
    """
    scaler = StandardScaler()
    data_std = scaler.fit_transform(data)
    pca = PCA(n_components=1)
    pca.fit(data_std)
    explained_var = pca.explained_variance_ratio_[0]
    complexity = 1 - explained_var
    return complexity

def calculate_cosine_similarities(data):
    """
    计算每行数据与平均值的余弦相似度
    
    参数:
        data: 二维numpy数组，形状为(nums, features)
        
    返回:
        余弦相似度列表，长度为nums，取值范围[-1, 1]，1表示完全相同，-1表示完全相反
    """
    mean_vector = np.mean(data, axis=0)
    dot_products = np.sum(data * mean_vector, axis=1)
    data_norms = np.sqrt(np.sum(data**2, axis=1))
    mean_norm = np.sqrt(np.sum(mean_vector**2))
    cosine_similarities = np.divide(
        dot_products,
        data_norms * mean_norm,
        out=np.zeros_like(dot_products, dtype=float),
        where=(data_norms * mean_norm) != 0
    )
    return cosine_similarities

def calculate_mahalanobis_distances(data):
    """
    计算每行数据与平均值的马氏距离
    
    参数:
        data: 二维numpy数组，形状为(nums, features)
        
    返回:
        马氏距离数组，长度为nums
    """
    mean_vector = np.mean(data, axis=0)
    cov_matrix = np.cov(data, rowvar=False)
    inv_cov_matrix = pinv(cov_matrix)  # 使用伪逆代替常规逆
    diff = data - mean_vector
    mahalanobis_dist = np.sqrt(np.sum(diff @ inv_cov_matrix * diff, axis=1))
    return mahalanobis_dist

def calculate_euclidean_distances(data):
    """
    计算每行数据与平均值的欧式距离
    
    参数:
        data: 二维numpy数组，形状为(nums, features)
        
    返回:
        欧式距离列表，长度为nums
    """
    mean_values = np.mean(data, axis=0)
    distances = np.sqrt(np.sum((data - mean_values)**2, axis=1)) 
    return distances  # 转换为列表返回