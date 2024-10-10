import numpy as np
from sklearn.datasets import make_spd_matrix

from probmlpy import normal_impute

np.random.seed(12)
data_dim = 8
n_data = 10
rate_missing = 0.5

mu = np.random.randn(data_dim)
sigma = make_spd_matrix(n_dim=data_dim)  # Generate a random positive semi-definite matrix
# test if the matrix is positive definite
# print(is_pos_def(sigma))

x_full = np.random.multivariate_normal(mu, sigma, n_data)
missing = np.random.choice([0, 1], size=10, p=[1-rate_missing, rate_missing])

x_miss = np.copy(x_full)
x_miss[missing] = np.nan

x_imputed = normal_impute(mu, sigma, x_miss)

