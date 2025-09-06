import numpy as np
import torch
import torch.nn as nn
from statsmodels.tsa.stattools import ccovf 
from statsmodels.tsa.stattools import acovf
import math


def gamma_l(center_dat: torch.Tensor, lag: int) -> torch.Tensor:
    d, J = center_dat.shape
    gamma_lag_sum = torch.zeros((J, J), dtype=center_dat.dtype, device=center_dat.device)

    if lag >= 0:
        for ij in range(d - lag):
            x = center_dat[ij].unsqueeze(1)         # shape (J, 1)
            y = center_dat[ij + lag].unsqueeze(1)   # shape (J, 1)
            gamma_lag_sum += x @ y.T                # outer product
    else:
        for ij in range(d + lag):
            x = center_dat[ij - lag].unsqueeze(1)
            y = center_dat[ij].unsqueeze(1)
            gamma_lag_sum += x @ y.T

    return gamma_lag_sum / d


def kweights(u: torch.Tensor, kernel: str) -> torch.Tensor:
    """
    u: normalized argument (e.g., lag / bandwidth), same shape tensor
    kernel: one of {"Bartlett","Parzen","TH","QS"}
    """
    # k = kernel.lower()
    k = kernel
    abs_u = torch.abs(u)

    if k == "Bartlett":
        return torch.clamp(1 - abs_u, min=0.0)
    elif k == "parzen":
        w = torch.where(
            abs_u <= 0.5,
            1 - 6 * abs_u**2 + 6 * abs_u**3,
            torch.where(
                abs_u <= 1.0,
                2 * (1 - abs_u) ** 3,
                torch.zeros_like(u),
            ),
        )
        return w
    elif k == "TH":
        # W_TH(x) = (1 + cos(pi x))/2 for |x| <= 1; 0 otherwise
        w = 0.5 * (1.0 + torch.cos(math.pi * u))
        return torch.where(abs_u <= 1.0, w, torch.zeros_like(u))
    elif k == "QS":
        # W_QS(x) = 25/(12 π^2 x^2) * ( sin(6πx/5)/(6πx/5) - cos(6πx/5) )
        z = (6.0 * math.pi / 5.0) * u
        # use torch.sinc for stability: sinc(y) = sin(πy)/(πy) ⇒ sin(z)/z = sinc(z/π)
        sinc_term = torch.sinc(z / math.pi)
        term = sinc_term - torch.cos(z)
        eps = math.sqrt(torch.finfo(u.dtype).eps)
        inv_u2 = torch.where(abs_u > eps, 1.0 / (u * u), torch.zeros_like(u))
        w = (25.0 / (12.0 * math.pi**2)) * inv_u2 * term
        # define the removable singularity at 0 by continuity: W_QS(0) = 1
        w = torch.where(abs_u > eps, w, torch.ones_like(u))
        return w

    elif k == "flattop":
        return flat_top_kernel(u)

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    
def flat_top_kernel(u: torch.Tensor) -> torch.Tensor:
    abs_u = torch.abs(u)
    out = torch.zeros_like(abs_u)
    out[abs_u < 0.5] = 1.0
    mask = (abs_u >= 0.5) & (abs_u <= 1)
    out[mask] = 2 - 2 * abs_u[mask]
    return out


def compute_gamma_hat(center_dat: torch.Tensor) -> list:
    d, T = center_dat.shape
    return [gamma_l(center_dat, lag=ik) for ik in range(T)]


def cov_l(center_dat: torch.Tensor, porder: int, band: int, kern_type: str):
    d, J = center_dat.shape

    cov_weighted = gamma_l(center_dat, 0).clone()
    cov_unweighted = gamma_l(center_dat, 0).clone()

    gamma_m1 = gamma_l(center_dat, -1)
    gamma_0 = gamma_l(center_dat, 0)
    gamma_p1 = gamma_l(center_dat, 1)
    cov_triple = gamma_m1 + gamma_0 + gamma_p1

    for ik in range(1, d):
        u = ik / band
        if u > 1:
            break
        weight = kweights(torch.tensor([u]), kernel=kern_type)
        gamma_k = gamma_l(center_dat, ik)
        term = (abs(ik) ** porder) * (gamma_k + gamma_k.T)
        cov_weighted += weight * term
        cov_unweighted += term

    return cov_weighted, cov_unweighted, cov_triple


class Model1_new(nn.Module):
    def __init__(self, x_shape_0, x_shape_1, cov_dim, hidden_dim, y_shape_0, y_shape_1,
                 cov_matrix: torch.Tensor):
        super(Model1_new, self).__init__()
        self.hidden_dim = hidden_dim
        self.y_shape_1 = y_shape_1

        # Save precomputed covariance matrix (J x J)
        self.register_buffer('cov_matrix', cov_matrix)

        # Input layers: X (J) + interaction (J) → 2J
        self.fc1 = nn.Linear(x_shape_1 + cov_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, self.y_shape_1)

        self.activation = nn.GELU()

        # Projection from d (timepoints) to y_shape_0
        self.proj = nn.Parameter(torch.randn(y_shape_0, x_shape_0))  # (y_shape_0, d)

    def forward(self, X):  # X: shape (d, J)
        # Use precomputed covariance
        cov_interaction = X @ self.cov_matrix.T  # (d, J)

        # Concatenate X and interaction
        combined_input = torch.cat((X, cov_interaction), dim=1)  # (d, 2J)

        # Forward through the network
        out = self.activation(self.fc1(combined_input))
        out = self.activation(self.fc2(out))
        out = self.fc3(out)  # (d, y_shape_1)

        # Project across curves
        out = self.proj @ out  # (y_shape_0, y_shape_1)
        return out
