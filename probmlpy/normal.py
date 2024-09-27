import numpy as np

def normal_condition(mu, sigma, visible_nodes, visible_values):
    """
    根据多元正态分布计算条件分布。
    
    给定部分可见变量的值，计算其余未见变量（隐藏变量）的条件均值和协方差矩阵。

    参数：
    - mu: 多元正态分布的均值向量 (1D numpy 数组)
    - sigma: 多元正态分布的协方差矩阵 (2D numpy 数组)
    - visible_nodes: 已知（可见）变量的索引 (1D numpy 数组或列表)
    - visible_values: 已知（可见）变量的具体取值 (1D numpy 数组)

    返回：
    - mugivh: 条件均值向量 (1D numpy 数组)
    - sigivh: 条件协方差矩阵 (2D numpy 数组)
    """
    
    d = len(mu)  # 正态分布的维度
    j = np.arange(d)  # 所有变量的索引列表 [0, 1, ..., d-1]
    v = np.asarray(visible_nodes).reshape(-1)  # 将可见变量索引转换为1D数组
    h = np.setdiff1d(j, v)  # 未知（隐藏）变量的索引集，等于所有变量索引减去可见变量索引

    # 如果没有隐藏变量，返回空数组
    if len(h) == 0:
        return np.array([]), np.array([])  
    # 如果没有可见变量，返回原始的均值和协方差
    elif len(v) == 0:
        return mu, sigma  

    # 提取协方差矩阵中涉及到隐藏变量和可见变量的子矩阵：
    sigma_hh = sigma[np.ix_(h, h)]  # 隐藏变量之间的协方差矩阵
    sigma_hv = sigma[np.ix_(h, v)]  # 隐藏变量与可见变量之间的协方差矩阵
    sigma_vv = sigma[np.ix_(v, v)]  # 可见变量之间的协方差矩阵
    
    # 计算可见变量协方差矩阵的逆，用于计算条件分布
    sigma_vv_inv = np.linalg.solve(sigma_vv, np.eye(len(v)))  # 使用solve代替显式求逆

    # 计算隐藏变量的条件均值
    delta_mu = visible_values - mu[v]  # 可见变量的偏移量，即 (x_v - mu_v)
    mugivh = mu[h] + sigma_hv @ sigma_vv_inv @ delta_mu  # 条件均值公式： mu_h + Σ_hv Σ_vv^{-1} (x_v - mu_v)

    # 计算隐藏变量的条件协方差
    sigivh = sigma_hh - sigma_hv @ sigma_vv_inv @ sigma_hv.T  # 条件协方差公式： Σ_hh - Σ_hv Σ_vv^{-1} Σ_hv^T

    return mugivh, sigivh





def normal_impute(mu, sigma, x):
    """
    对数据矩阵 x 中的缺失值 (NaN) 进行正态分布插补。
    
    参数:
    - mu: 多元正态分布的均值向量 (1D numpy 数组)
    - sigma: 多元正态分布的协方差矩阵 (2D numpy 数组)
    - x: 含有缺失值 (NaN) 的数据矩阵 (2D numpy 数组)
    
    返回:
    - x_imputed: 插补后的数据矩阵 (2D numpy 数组)
    """
    x_imputed = np.copy(x)
    n_data = x.shape[0]
    
    for i in range(n_data):
        row = x[i, :]
        hidden_mask = np.isnan(row)  # 布尔掩码，标记隐藏（缺失）值的位置
        visible_mask = ~hidden_mask  # 布尔掩码，标记可见（已知）值的位置
        
        if np.any(hidden_mask):  # 如果存在缺失值
            visible_nodes = np.where(visible_mask)[0]  # 获取可见变量的索引
            visible_values = row[visible_mask]  # 提取可见变量的值
            
            # 通过可见变量计算隐藏变量的条件分布
            mu_hgv, _ = normal_condition(mu, sigma, visible_nodes, visible_values)
            
            # 用条件均值插补缺失值
            x_imputed[i, hidden_mask] = mu_hgv
    
    return x_imputed


