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
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm 

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
    seed_offset: int = 0,
    max_workers: int = 1,
):
    if band is None:
        band = round(d_large ** (1 / 3))

    cov_matrices = []

    def run_one(i):
        seed = seed_offset + i

        if Theta1 is None and Theta2 is None:
            X_coef, F_basis, fd_basis = generate_far2_coef(
                d=d_large, J=J, Psi1=Phi1, Psi2=Phi2, Sigma=Sigma, seed=seed
            )
        else:
            X_coef, F_basis, fd_basis = generate_farma22_coef(
                d=d_large, J=J,
                Phi1=Phi1, Phi2=Phi2,
                Theta1=Theta1, Theta2=Theta2,
                Sigma=Sigma, seed=seed
            )

        fpca_fbasis = FPCA(n_components=n_components)
        fpca_fbasis.fit(fd_basis)
        scores = fpca_fbasis.transform(fd_basis)

        scores_centered = scores - scores.mean(axis=0, keepdims=True)
        X = torch.tensor(scores_centered, dtype=torch.float32)

        with torch.no_grad():
            cov_weighted, _, _ = cov_l(X, porder=0, band=band, kern_type=kern_type)

        return cov_weighted.cpu().numpy()

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(run_one, i) for i in range(N)]
            for fut in tqdm(as_completed(futures), total=N, desc="Simulations"):
                cov_matrices.append(fut.result())
    else:
        for i in tqdm(range(N), desc="Simulations"):
            cov_matrices.append(run_one(i))

    empirical_truth = np.mean(np.stack(cov_matrices, axis=0), axis=0)
    return empirical_truth


# Ensure folder exists
os.makedirs("true_LRcov", exist_ok=True)

J = 5
N = 500
d_large = 100000
kernel_list = ["Bartlett","Parzen","TH", "QS"]

for kernel in kernel_list:
    print(kernel)
    Sigma = 1.0 / np.arange(1, J + 1)
    psi1 = 0.5 * np.eye(J)
    psi2 = -0.2 * np.eye(J)

    if kernel == "Bartlett":
        band_list = [5,4,2]
    else:
        band_list = [5,4,3,2]

    for bandtype in band_list:
        band = round(d_large ** (1 / bandtype))
        print(band)

        empirical_cov_far2 = estimate_empirical_truth_farma(
            N=N, d_large=d_large, J=J, n_components=J,
            Phi1=psi1, Phi2=psi2, Sigma=Sigma,
            band=band, kern_type=kernel, seed_offset=0,
            max_workers=64   # parallel with progress bar
        )

        filename = f"true_LRcov/empirical_cov_far2_N{N}_d{d_large}_band{band}_kernel{kernel}.csv"
        np.savetxt(filename, empirical_cov_far2, delimiter=",")
        print(f"Saved to: {filename}")
