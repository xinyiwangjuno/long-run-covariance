import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
import pandas as pd
from scipy.integrate import quad
from skfda.preprocessing.dim_reduction import FPCA
import torch
import torch.nn as nndata
import torch.optim as optim
from models_HLS import *
from data_generation_new import *
from utils import *
from training_HLS import *
import os


def estimate_empirical_truth_farma(
    N: int,
    d_large: int,
    J: int,
    n_components: int,
    Phi1=None,
    Phi2=None,
    Theta1=None,
    Theta2=None,
    Sigma=None,
    band: int = None,
    kern_type: str = "Bartlett",
    seed_offset: int = 0
):
    """
    Estimate empirical long-run covariance from simulated FAR(2) or FARMA(2,2) data.

    Parameters:
        N: number of simulations
        d_large: number of time points per simulation
        J: number of basis functions
        n_components: number of FPCA components
        Phi1, Phi2: AR operators
        Theta1, Theta2: MA operators (if None, assumes FAR(2))
        Sigma: std devs for white noise
        band: bandwidth for kernel estimation (if None, uses rule-of-thumb)
        kern_type: kernel type ('Bartlett', 'FlatTop', etc.)
        seed_offset: random seed offset

    Returns:
        np.ndarray: averaged long-run covariance matrix (n_components x n_components)
    """
    if band is None:
        band = round(d_large ** (1 / 3))

    cov_matrices = []

    for i in range(N):
        seed = seed_offset + i

        if Theta1 is None and Theta2 is None:
            # FAR(2)
            X_coef, F_basis, fd_basis = generate_far2_coef(
                d=d_large, J=J, Psi1=Phi1, Psi2=Phi2, Sigma=Sigma, seed=seed
            )
        else:
            # FARMA(2,2)
            X_coef, F_basis, fd_basis = generate_farma22_coef(
                d=d_large, J=J,
                Phi1=Phi1, Phi2=Phi2,
                Theta1=Theta1, Theta2=Theta2,
                Sigma=Sigma, seed=seed
            )

        # FPCA
        fpca_fbasis = FPCA(n_components=n_components)
        fpca_fbasis.fit(fd_basis)
        scores = fpca_fbasis.transform(fd_basis)

        # Center the scores
        scores_centered = scores - scores.mean(axis=0, keepdims=True)
        X = torch.tensor(scores_centered, dtype=torch.float32)

        # Long-run covariance
        cov_weighted, _, _ = cov_l(X, porder=0, band=band, kern_type=kern_type)
        cov_matrices.append(cov_weighted.numpy())

    empirical_truth = np.mean(np.stack(cov_matrices, axis=0), axis=0)
    return empirical_truth




# Ensure folder exists
os.makedirs("true_LRcov", exist_ok=True)

# Parameters
J = 5
N = 500
d_large = 100000
kernel_list = ["Bartlett","Parzen","TH", "QS"]
for kernel in kernel_list:
    print(kernel)
    Sigma = 1.0 / np.arange(1, J + 1)
    psi1 = 0.5 * np.eye(J)
    psi2 = -0.2 * np.eye(J)
    #band_list = [2,3,4,5]
    if kernel == "Barlett":
        band_list = [2,4,5]
    else:
        band_list = [2,3,4,5]

    for bandtype in band_list:
        # Compute bandwidth
        band = round(d_large ** (1 / bandtype))
        print(band)
        # Estimate long-run covariance
        empirical_cov_far2 = estimate_empirical_truth_farma(
            N=N, d_large=d_large, J=J, n_components=J,
            Phi1=psi1,
            Phi2=psi2,
            Sigma=Sigma,
            band = band,
            kern_type=kernel,
            seed_offset=0
        )

        # Save
        # Save path
        filename = f"true_LRcov/empirical_cov_far2_N{N}_d{d_large}_band{band}_kernel{kernel}.csv"

        # Save the matrix
        np.savetxt(filename, empirical_cov_far2, delimiter=",")
        print(f"Saved to: {filename}")