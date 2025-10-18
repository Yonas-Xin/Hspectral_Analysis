"""从spectral库中重新封装算法，对部分算法做了调整"""
import spectral as spy
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.linalg import pinv
from tqdm import tqdm
# import torch
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

def smacc(spectra, min_endmembers=None, max_residual_norm=float('Inf')):
    '''
    Returns SMACC decomposition and endmember binary mask.
    
    Arguments:
        `spectra` (ndarray): 2D (N x B) or 3D (rows x cols x B) spectral data
        `min_endmembers`: Minimum number of endmembers
        `max_residual_norm`: Residual norm threshold for stopping
    
    Returns:
        S: Endmember spectra (num_endmembers x B)
        F: Abundance coefficients (N x num_endmembers)
        R: Residual matrix (N x B)
        endmember_mask: Binary mask where 1 indicates endmember positions
                       Shape is (N,) for 2D input or (rows, cols) for 3D input
    '''
    # Initialize variables
    q = []  # Indices of endmembers
    input_was_3d = len(spectra.shape) == 3
    
    # Reshape if input is 3D
    if input_was_3d:
        rows, cols, bands = spectra.shape
        H = spectra.reshape((-1, bands))
        original_shape = (rows, cols)
    else:
        H = spectra
        original_shape = None
    
    R = H
    Fs = []
    endmember_mask = np.zeros(H.shape[0], dtype=int)  # Initialize flat mask
    
    # Set default min_endmembers
    if min_endmembers is None:
        min_endmembers = np.linalg.matrix_rank(H)
    
    # Initial residual norms
    residual_norms = np.sqrt(np.einsum('ij,ij->i', H, H))
    current_max_residual_norm = np.max(residual_norms)
    
    if max_residual_norm is None:
        max_residual_norm = current_max_residual_norm / min_endmembers
    pbar = tqdm(total=min_endmembers) # 进度条
    # Main SMACC loop
    while len(q) < min_endmembers or current_max_residual_norm > max_residual_norm:
        new_endmember_idx = np.argmax(residual_norms)
        q.append(new_endmember_idx)
        endmember_mask[new_endmember_idx] = 1  # Mark as endmember
        
        n = len(q) - 1
        w = R[q[n]]
        wt = w / (np.dot(w, w))
        On = np.dot(R, wt)
        alpha = np.ones(On.shape, dtype=np.float64)
        
        # Correct alphas for oblique projection
        for k in range(len(Fs)):
            t = On * Fs[k][q[n]]
            t[t == 0.0] = 1e-10
            np.minimum(Fs[k]/t, alpha, out=alpha)
        
        # Clip negative coefficients
        alpha[On <= 0.0] = 0.0
        alpha[q[n]] = 1.0
        
        # Calculate oblique projection coefficients
        Fn = alpha * On
        Fn[Fn <= 0.0] = 0.0
        
        # Update residual
        R = R - np.outer(Fn, w)
        
        # Update previous projection coefficients
        for k in range(len(Fs)):
            Fs[k] -= Fs[k][q[n]] * Fn
            Fs[k][Fs[k] <= 0.0] = 0.0
        
        Fs.append(Fn)
        pbar.update(1)
        # Update residual norms
        residual_norms[:] = np.sqrt(np.sum(R * R, axis=1))  # 替代 np.einsum
        current_max_residual_norm = np.max(residual_norms)
        # print(f'Found {len(q)} endmembers, current max residual norm is {current_max_residual_norm:.4f}\r', end='')
    
    # Final correction as suggested in SMACC paper
    for k, s in enumerate(q):
        Fs[k][q] = 0.0
        Fs[k][s] = 1.0
    
    F = np.array(Fs).T
    S = H[q]
    
    # Reshape mask if input was 3D
    if input_was_3d:
        endmember_mask = endmember_mask.reshape(original_shape)
    return S, F, R, endmember_mask

# def smacc_gpu(spectra_np, min_endmembers=None, max_residual_norm=float('inf')):
#     '''
#     SMACC算法：NumPy输入输出 + GPU计算（PyTorch 实现）

#     输入:
#         spectra_np: NumPy数组，形状 (N, B) 或 (rows, cols, B)
#         min_endmembers: 最小端元数
#         max_residual_norm: 最大残差范数，超过则继续提取端元

#     返回:
#         S: NumPy数组，端元光谱 (num_endmembers x B)
#         F: NumPy数组，丰度系数 (N x num_endmembers)
#         R: NumPy数组，残差矩阵 (N x B)
#         endmember_mask: NumPy数组，端元掩码，形状 (N,) 或 (rows, cols)
#     '''
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     input_was_3d = spectra_np.ndim == 3
#     if input_was_3d:
#         rows, cols, bands = spectra_np.shape
#         H = torch.from_numpy(spectra_np.reshape(-1, bands)).float().to(device)
#         original_shape = (rows, cols)
#     else:
#         H = torch.from_numpy(spectra_np).float().to(device)
#         original_shape = None

#     R = H.clone()
#     q = []
#     Fs = []
#     endmember_mask = torch.zeros(H.shape[0], dtype=torch.int32, device=device)

#     if min_endmembers is None:
#         min_endmembers = torch.linalg.matrix_rank(H).item()

#     residual_norms = torch.norm(H, dim=1)
#     current_max_residual_norm = torch.max(residual_norms)

#     if max_residual_norm is None:
#         max_residual_norm = current_max_residual_norm / min_endmembers

#     while len(q) < min_endmembers or current_max_residual_norm > max_residual_norm:
#         new_endmember_idx = torch.argmax(residual_norms).item()
#         q.append(new_endmember_idx)
#         endmember_mask[new_endmember_idx] = 1

#         w = R[q[-1]]
#         wt = w / torch.dot(w, w)
#         On = torch.matmul(R, wt)
#         alpha = torch.ones_like(On, dtype=torch.float32)

#         for k in range(len(Fs)):
#             t = On * Fs[k][q[-1]]
#             t = torch.where(t == 0.0, torch.tensor(1e-10, device=device), t)
#             alpha = torch.minimum(Fs[k] / t, alpha)

#         alpha = torch.where(On <= 0.0, torch.tensor(0.0, device=device), alpha)
#         alpha[q[-1]] = 1.0

#         Fn = alpha * On
#         Fn = torch.where(Fn <= 0.0, torch.tensor(0.0, device=device), Fn)

#         R -= torch.ger(Fn, w)

#         for k in range(len(Fs)):
#             Fs[k] -= Fs[k][q[-1]] * Fn
#             Fs[k] = torch.where(Fs[k] <= 0.0, torch.tensor(0.0, device=device), Fs[k])

#         Fs.append(Fn)
#         residual_norms = torch.norm(R, dim=1)
#         current_max_residual_norm = torch.max(residual_norms)
#         print(f'Found {len(q)} endmembers, current max residual norm is {current_max_residual_norm:.4f}\r', end='')

#     for k, s in enumerate(q):
#         Fs[k][q] = 0.0
#         Fs[k][s] = 1.0

#     F = torch.stack(Fs, dim=1)
#     S = H[q]

#     if input_was_3d:
#         endmember_mask = endmember_mask.reshape(original_shape)

#     # 返回值全部转回 numpy
#     return S.cpu().numpy(), F.cpu().numpy(), R.cpu().numpy(), endmember_mask.cpu().numpy()
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