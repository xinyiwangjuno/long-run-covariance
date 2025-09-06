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
from tqdm import tqdm
import os

ds = [500, 1000, 2000, 4000, 8000]
band_types = ["2","3","4","5"]
band_type_map = {
    "2": lambda d: round(d ** (1/2)),
    "3": lambda d: round(d ** (1/3)),
    "4": lambda d: round(d ** (1/4)),
    "5": lambda d: round(d ** (1/5)),
    "6": lambda d: round(d ** (1/6)),
    "7": lambda d: round(d ** (1/7)),
    "8": lambda d: round(d ** (1/8)),
    "fix_2": lambda d: 2,
}
model = "FMA2" # model = "FMA1"
kern_type = "QS" # Choose from "Bartlett","Parzen","TH","QS"
for d in ds:    
    for band_type in band_types:
        # Parameters
        J = 5
        num_components = 5
        band = band_type_map[band_type](d)
        print(band)
        sigma = 1.0 / np.arange(1, J + 1)
        porder = 0
        nrepli = 1000

        if model == "FMA1":
            theta1 = 0.5 * np.eye(J)
            theta2 = None
            # Theoretical long-run covariance
            Sigma2 = np.diag(sigma**2)
            g0 = Sigma2 + theta1 @ Sigma2 @ theta1.T 
            g1 = theta1 @ Sigma2
            g_1 = Sigma2 @ theta1.T
            theoretical_cov_true = g0 + g1 + g_1 

        # Theoretical long-run covariance
        if model == "FMA2":
            theta1 = 0.8 * np.eye(J)
            theta2 = 0.8 * np.eye(J)
            Sigma2 = np.diag(sigma**2)
            g0 = Sigma2 + theta1 @ Sigma2 @ theta1.T + theta2 @ Sigma2 @ theta2.T
            g1 = theta1 @ Sigma2 + theta2 @ Sigma2 @ theta1.T
            g2 = theta2 @ Sigma2
            g_1 = Sigma2 @ theta1.T + theta1 @ Sigma2 @ theta2.T
            g_2 = Sigma2 @ theta2.T
            theoretical_cov_true = g0 + g1 + g2 + g_1 + g_2

        # Storage
        squared_norms = []

        # Loop for multiple seeds
        for seed in tqdm(range(nrepli)):
            # Generate data
            X_coef, F_basis, fd_basis = generate_fma2_coef(
                J=J, d=d, Sigma=sigma, Theta1=theta1, Theta2=theta2, seed=seed
            )

            # FPCA
            fpca_fbasis = FPCA(n_components=num_components)
            fpca_fbasis.fit(fd_basis)
            scores = fpca_fbasis.transform(fd_basis)
            E_coef = fpca_fbasis.components_.coefficients
            center_dat = scores

            # Estimate long-run covariance
            X = torch.tensor(center_dat, dtype=torch.float32)
            cov_weighted, _, _ = cov_l(X, porder, band, kern_type)
            cov_weighted_np = E_coef.T @ cov_weighted.detach().numpy() @ E_coef

            # Riemann sum approximation of squared HS norm
            diff = cov_weighted_np - theoretical_cov_true
            delta = 1.0 / diff.shape[0]
            squared_norm = np.sum(diff**2) * delta**2
            squared_norms.append(squared_norm)

        # Get current script directory
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Create 'results' subfolder in the current directory
        results_dir = os.path.join(current_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        # Construct the full file path
        filename = f"{model}_d{d}_J{J}_numcomp{num_components}_band{band_type}_kernel{kern_type}.csv"
        file_path = os.path.join(results_dir, filename)

        # Save the CSV
        pd.DataFrame({"squared_norm": squared_norms}).to_csv(file_path, index=False)
        print(f"Saved to {file_path}")   
