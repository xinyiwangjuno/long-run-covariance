import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import ccovf 
from statsmodels.tsa.stattools import acovf


# Define a small neural network to estimate the weight function

class Weight_Barlett(nn.Module):
    def __init__(self, m_init, q_init):
        super(Weight_Barlett, self).__init__()
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

class Weight_Parzen(nn.Module):
    def __init__(self, m_init):
        super(Weight_Parzen, self).__init__()
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))

    def forward(self, x):
        x_scaled = torch.abs(x / self.m)
        W = torch.zeros_like(x_scaled)
        mask1 = x_scaled <= 0.5
        mask2 = (x_scaled > 0.5) & (x_scaled <= 1)
        W[mask1] = 1 - 6 * x_scaled[mask1]**2 + 6 * x_scaled[mask1]**3
        W[mask2] = 2 * (1 - x_scaled[mask2])**3
        return W

class Weight_TH(nn.Module):
    def __init__(self, m_init):
        super(Weight_TH, self).__init__()
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))

    def forward(self, x):
        x_scaled = x / self.m
        x_abs = torch.abs(x_scaled)
        W = torch.zeros_like(x_scaled)
        mask = x_abs <= 1
        W[mask] = 0.5 * (1 + torch.cos(torch.pi * x_scaled[mask]))
        return W

class Weight_QS(nn.Module):
    def __init__(self, m_init):
        super(Weight_QS, self).__init__()
        self.m = nn.Parameter(torch.tensor(m_init, dtype=torch.float32))

    def forward(self, x):
        x_scaled = x / self.m
        x_abs = torch.abs(x_scaled)
        W = torch.zeros_like(x_scaled)
        mask = x_abs > 1e-6  # Avoid division by zero
        t = 6 * torch.pi * x_scaled[mask] / 5
        W[mask] = (25 / (12 * torch.pi**2 * x_scaled[mask]**2)) * \
                  ((torch.sin(t) / t) - torch.cos(t))
        W[~mask] = 1.0  # Define W(0) = 1
        return W



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
    weights = weight_net(lags)  # Compute weights using the weight network
    #weights = weights / weights.sum()
    #print(weights.size()
    #print(weights)
    #print(weights.sum())

    for i in range(n_components):
        for j in range(n_components):
            # Extract scores for the i-th and j-th components
            score_i = X[i, :].cpu().numpy()  # Convert to NumPy for ccovf
            score_j = X[j, :].cpu().numpy()  # Convert to NumPy for ccovf
            
            # Compute cross-covariance using ccovf
            cross_cov = ccovf(score_i.T, score_j.T, adjusted=True, demean=True, fft=True)
            # Truncate cross-covariance to the range of lags (-truncation_q to truncation_q)
            cross_cov_truncated = np.concatenate([
                cross_cov[1 : truncation_q + 1][::-1],  # Negative lags (-truncation_q to -1)
                cross_cov[: truncation_q + 1]           # Positive lags (0 to truncation_q)
            ])
#           
            cross_cov_truncated = torch.tensor(cross_cov_truncated, dtype=weights.dtype, device=weights.device)
            weighted_cross_cov = cross_cov_truncated * weights
            #print(cross_cov_truncated)
            
            # Sum the weighted cross-covariances to get the long-run covariance
            cov_matrix[i, j] = torch.tensor(torch.sum(weighted_cross_cov), dtype=X.dtype, device=X.device)
            cov_matrix_w1[i, j] = torch.tensor(torch.sum(cross_cov_truncated), dtype=X.dtype, device=X.device)
            
            # Compute cov_matrix3: sum of gamma(-1), gamma(0), and gamma(1)
            gamma_0 = cross_cov[0]  # Lag 0
            gamma_1 = cross_cov[1]  # Lag 1
            gamma_neg1 = cross_cov[1]  # Lag -1
            cov_matrix3[i, j] = torch.tensor(gamma_0 + gamma_1 + gamma_neg1, dtype=X.dtype, device=X.device)
    
    return cov_matrix, cov_matrix_w1, cov_matrix3




# Neural network model using interaction terms between X and the covariance matrix

class Model1(nn.Module):
    def __init__(self, x_shape_0, x_shape_1, cov_dim, hidden_dim, y_shape_0, y_shape_1, weight_net, trunc_q):
        super(Model1, self).__init__()
        self.weight_net = weight_net  # Store weight network
        self.trunc_q = trunc_q # Store q parameter for LR_cov_X
        self.hidden_dim = hidden_dim
        self.y_shape_1 = y_shape_1  # Store y_shape_1 for reshaping

        self.fc1 = nn.Linear(x_shape_1 + cov_dim, hidden_dim)  # Fix input size
        self.fc2 = nn.Linear(self.hidden_dim, self.y_shape_1)  # Ensure correct output size
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()

        self.proj = nn.Parameter(torch.randn(y_shape_0, x_shape_0)) 

    def forward(self, X):
        # Compute covariance matrix **inside forward**
        cov_matrix, _, _ = LR_cov_X(X, self.weight_net, self.trunc_q)  

        # Interaction between X and covariance matrix
        cov_interaction = cov_matrix @ X  # Matrix multiplication
        combined_input = torch.cat((X, cov_interaction), dim=1)  # Concatenate X and cov_interaction
        #print(f"Combined Input Shape: {combined_input.shape}")

        # Pass through neural network layers
        out = self.activation1(self.fc1(combined_input))  # Shape: (x_shape_0, hidden_dim)
        #print(out.shape)
        out = self.fc2(out)  # Shape: (x_shape_0, y_shape_1)
        #print(out.shape)
        out = self.activation2(out)
        #print(f"Output Shape Before Reshape: {out.shape}")

        # Apply projection (output remains (hidden_dim, y_shape_1))
        out = self.proj @ out  

        #print(f"Final Output Shape: {out.shape}")  # Debugging

        return out


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

class Model1_MHA(nn.Module):
    def __init__(self, x_shape_0, x_shape_1, cov_dim, hidden_dim, y_shape_0, y_shape_1, weight_net, trunc_q, num_heads=4):
        super(Model1_MHA, self).__init__()
        self.weight_net = weight_net  
        self.trunc_q = trunc_q  
        self.hidden_dim = hidden_dim
        self.y_shape_1 = y_shape_1  

        # Multi-Head Attention Layer
        self.mha = nn.MultiheadAttention(embed_dim=x_shape_1, num_heads=num_heads, batch_first=True)
        
        # Fully connected layers after attention
        self.fc1 = nn.Linear(x_shape_1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.y_shape_1)

        self.activation = nn.GELU()
        
        # Projection matrix
        self.proj = nn.Parameter(torch.randn(y_shape_0, x_shape_0))  

    def forward(self, X):
        # Compute covariance matrix **inside forward**
        cov_matrix, _, _ = LR_cov_X(X, self.weight_net, self.trunc_q)  

        # Compute interaction
        cov_interaction = cov_matrix @ X  # Shape: (n_components, d)

        # Apply Multi-Head Attention
        attn_output, _ = self.mha(X, cov_interaction, cov_interaction)  # Query=X, Key=Value=cov_interaction

        # Pass through fully connected layers
        out = self.activation(self.fc1(attn_output))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)
        
        # Apply projection
        out = self.proj @ out  

        return out


# Neural network model without covariance matrix interaction
class Model1_nocov(nn.Module):
    def __init__(self, x_shape_0, x_shape_1, hidden_dim, y_shape_0, y_shape_1):
        super(Model1_nocov, self).__init__()
        self.fc1 = nn.Linear(x_shape_1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, y_shape_1)
        self.activation1 = nn.GELU()
        self.activation2 = nn.GELU()
        self.proj = nn.Parameter(torch.randn(y_shape_0, x_shape_0))

    def forward(self, X):
        # Pass X directly through the network layers
        out = self.activation1(self.fc1(X))
        out = self.fc2(out)
        out = self.activation2(out)
        out = self.proj @ out

        return out

