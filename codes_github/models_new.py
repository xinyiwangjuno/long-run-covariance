import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import ccovf 
from statsmodels.tsa.stattools import acovf

class WeightNetwork(nn.Module):
    def __init__(self, m_init, q_init):
        super(WeightNetwork, self).__init__()
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))  # Trainable m
        self.q = nn.Parameter(torch.tensor(q_init, dtype=torch.float32))  # Trainable q
        #print("weight_net.m.requires_grad:", self.m.requires_grad)
        #print("weight_net.q.requires_grad:", self.q.requires_grad)

    def forward(self, x):
        x_abs = torch.abs(x)  # Ensure symmetry
        #W_q = torch.clamp(1 - (x_abs / self.m) ** self.q, min=0)  # Apply weight function
        W_q = torch.clamp(1 - (x_abs/self.m)**self.q, min=0) 
        #print(f"x_abs {x_abs}, W_q {W_q}")
        return W_q

    
def manual_cross_covariance(x, y, max_lag, demean=True):
    """
    Compute cross-covariances gamma_l for l in [-max_lag, ..., 0, ..., max_lag]
    using the formula in the screenshot. Returns a NumPy array of length 2*max_lag+1.
    
    Parameters:
    -----------
    x, y : 1D NumPy arrays of the same length n
    max_lag : integer, the maximum lag (positive or negative)
    demean : bool, whether to subtract means from x and y
    
    Returns:
    --------
    cross_cov : 1D NumPy array of length 2*max_lag + 1
                The element cross_cov[k] corresponds to gamma_{ell},
                where ell = k - max_lag. 
                i.e. cross_cov[0] = gamma_{-max_lag}, ..., 
                     cross_cov[max_lag] = gamma_0, 
                     cross_cov[2*max_lag] = gamma_{+max_lag}.
    """
    n = len(x)
    if demean:
        x_mean = np.mean(x)
        y_mean = np.mean(y)
    else:
        x_mean = 0.0
        y_mean = 0.0

    cross_cov = []
    # Loop over lags from -max_lag up to +max_lag
    for ell in range(-max_lag, max_lag + 1):
        s = 0.0
        
        if ell >= 0:
            # gamma_ell = (1/n) * sum_{t=1}^{n-ell} [ (x_t - x_mean)*(y_{t+ell} - y_mean) ]
            # Indexing in Python is 0-based, so we adjust accordingly:
            #   t runs from 0 to n - ell - 1
            for t in range(n - ell):
                s += (x[t] - x_mean) * (y[t + ell] - y_mean)
        else:
            # ell < 0
            # gamma_ell = (1/n) * sum_{t=1}^{n+ell} [ (x_{t-ell} - x_mean)*(y_t - y_mean) ]
            #   in 0-based indexing: t runs 0 to n + ell - 1
            #   x_{t-ell} => x[t - ell]
            for t in range(n + ell):
                s += (x[t - ell] - x_mean) * (y[t] - y_mean)
        
        cross_cov.append(s / n)
    
    return np.array(cross_cov)


def LR_cov_X(X, weight_net, truncation_q):
    """
    Compute the long-run covariance matrix and related matrices for the given scores.

    Parameters:
        X (torch.Tensor): Tensor of shape (n_components, d), where n_components is the number of components
                          after FPCA, and d is the number of curves.
        weight_net (WeightNetwork): An instance of the WeightNetwork class to compute the weights.
        truncation_q (int): Maximum lag to consider in the summation (truncation parameter).

    Returns:
        cov_matrix (torch.Tensor): Long-run covariance matrix weighted by the kernel function.
        cov_matrix_w1 (torch.Tensor): Long-run covariance matrix without weighting.
        cov_matrix3 (torch.Tensor): Covariance matrix including only lags -1, 0, and 1.
    """
    n_components, d = X.shape  # n_components: dim after FPCA, d: number of curves
    
    # Initialize the covariance matrices
    cov_matrix = torch.zeros((n_components, n_components), dtype=X.dtype, device=X.device)
    cov_matrix_w1 = torch.zeros((n_components, n_components), dtype=X.dtype, device=X.device)
    cov_matrix3 = torch.zeros((n_components, n_components), dtype=X.dtype, device=X.device)
    
    # Compute weights for lags -truncation_q to truncation_q
    lags = torch.arange(-truncation_q, truncation_q + 1, dtype=torch.float32, device=X.device)  # Lags from -truncation_q to truncation_q
    print(lags)
    weights = weight_net(lags)  
    print(weights)

    for i in range(n_components):
        for j in range(n_components):
            # Extract scores for the i-th and j-th components
            score_i = X[i, :].cpu().numpy()  
            score_j = X[j, :].cpu().numpy() 

            cross_cov_truncated = manual_cross_covariance(x=score_i, y=score_j, max_lag=truncation_q, demean=True)
            cross_cov_truncated = torch.tensor(cross_cov_truncated, dtype=weights.dtype, device=weights.device)
            weighted_cross_cov = cross_cov_truncated * weights
            
            # Sum the weighted cross-covariances to get the long-run covariance
            cov_matrix[i, j] = torch.tensor(torch.sum(weighted_cross_cov), dtype=X.dtype, device=X.device)
            cov_matrix_w1[i, j] = torch.tensor(torch.sum(cross_cov_truncated), dtype=X.dtype, device=X.device)
            
            # Compute cov_matrix3: sum of gamma(-1), gamma(0), and gamma(1)
            gamma_0 = cross_cov_truncated[0]  # Lag 0
            gamma_1 = cross_cov_truncated[1]  # Lag 1
            gamma_neg1 = cross_cov_truncated[1]  # Lag -1
            cov_matrix3[i, j] = torch.tensor(gamma_0 + gamma_1 + gamma_neg1, dtype=X.dtype, device=X.device)
    
    return cov_matrix, cov_matrix_w1, cov_matrix3



class Model1_new(nn.Module):
    def __init__(self, x_shape_0, x_shape_1, cov_dim, hidden_dim, y_shape_0, y_shape_1, weight_net, trunc_q):
        super(Model1_new, self).__init__()
        self.weight_net = weight_net  # Store weight network
        self.trunc_q = trunc_q  # Store q parameter for LR_cov_X
        self.hidden_dim = hidden_dim
        self.y_shape_1 = y_shape_1  # Store y_shape_1 for reshaping

        # Adjust input dimension to accommodate both X and cov_interaction
        self.fc1 = nn.Linear(x_shape_1 + cov_dim, hidden_dim)  # Expanded input
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  # Additional hidden layer
        self.fc3 = nn.Linear(hidden_dim, self.y_shape_1)  # Output layer

        self.activation = nn.GELU()

        self.proj = nn.Parameter(torch.randn(y_shape_0, x_shape_0))  # Projection matrix

    def forward(self, X):
        # Compute covariance matrix **inside forward**
        cov_matrix, _, _ = LR_cov_X(X, self.weight_net, self.trunc_q)  

        # Compute interaction between X and covariance matrix
        cov_interaction = cov_matrix @ X  # Shape: (n_components, d)

        # Concatenate X and cov_interaction
        combined_input = torch.cat((X, cov_interaction), dim=1)

        # Pass through neural network layers
        out = self.activation(self.fc1(combined_input))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        
        # Apply projection (output remains (y_shape_0, y_shape_1))
        out = self.proj @ out  

        return out



