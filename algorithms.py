"""从spectral库中重新封装算法，对部分算法做了调整"""
import spectral as spy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
