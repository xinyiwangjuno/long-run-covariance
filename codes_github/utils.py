import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

def compute_theoretical_gamma_far2(Psi1, Psi2, Sigma, q=20, tol=1e-8):
    """
    Compute Gamma(h) for h = 0, ..., q for a FAR(2) process using recursive Yule-Walker equations.

    Args:
        Psi1, Psi2: AR operator matrices (J x J)
        Sigma: innovation covariance (J x J)
        q: number of lags
        tol: early stopping if Gamma becomes negligible

    Returns:
        Gamma_list: list of length q+1 with Gamma(h)
    """
    J = Sigma.shape[0]
    Gamma = [np.zeros((J, J)) for _ in range(q + 1)]

    # Initial guesses (identity works; doesn't affect convergence)
    Gamma[1] = np.eye(J)
    Gamma[2] = np.eye(J)

    # Use Yule-Walker equation at h = 0 to get Gamma[0]
    Gamma[0] = Psi1 @ Gamma[1].T + Psi2 @ Gamma[2].T + Sigma

    # Now recursively compute Gamma(h) for h >= 3
    for h in range(3, q + 1):
        Gamma[h] = Psi1 @ Gamma[h - 1] + Psi2 @ Gamma[h - 2]
        if np.linalg.norm(Gamma[h], ord='fro') < tol:
            print(f"Early stop at h = {h} due to small norm")
            return Gamma[:h+1]

    return Gamma



def plot_matrices_3d(matrices, titles=None, figsize=(12, 10)):
    """
    Plots multiple matrices as 3D surface plots in a 2x2 grid with a fixed z-axis range [0, 4].

    Parameters:
    - matrices: List of 2D numpy arrays (shape must be the same for all matrices).
    - titles: List of titles for each subplot (optional).
    - figsize: Tuple defining the figure size.
    """
    if len(matrices) != 4:
        raise ValueError("This function requires exactly 4 matrices.")
    
    # Create a common X, Y meshgrid
    X, Y = np.meshgrid(np.arange(matrices[0].shape[0]), np.arange(matrices[0].shape[1]))

    fig, axes = plt.subplots(2, 2, figsize=figsize, subplot_kw={"projection": "3d"})

    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(4)]
    
    for ax, matrix, title in zip(axes.flat, matrices, titles):
        ax.plot_surface(X, Y, matrix, cmap="viridis")
        ax.set_zlim(0, 4)  # Set fixed z-axis range
        ax.set_title(title)

    plt.tight_layout()
    plt.show()
    
    return fig



def plot_matrices_2d(matrices, titles=None, figsize=(12, 10), cmap="viridis"):
    """
    Plots multiple matrices as 2D heatmaps in a 2x2 grid with color bars outside each subplot.

    Parameters:
    - matrices: List of 2D PyTorch tensors or NumPy arrays.
    - titles: List of titles for each subplot (optional).
    - figsize: Tuple defining the figure size.
    - cmap: Colormap for heatmaps.
    """
    if len(matrices) != 4:
        raise ValueError("This function requires exactly 4 matrices.")

    # Ensure all matrices are NumPy arrays
    converted_matrices = []
    for i, matrix in enumerate(matrices):
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy
        elif not isinstance(matrix, np.ndarray):
            raise TypeError(f"Matrix {i+1} is not a NumPy array or PyTorch tensor: {type(matrix)}")
        converted_matrices.append(matrix)

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    if titles is None:
        titles = [f"Matrix {i+1}" for i in range(4)]
    
    # Define the color bar range (0 to 8)
    vmin, vmax = 0, 8

    for ax, matrix, title in zip(axes.flat, converted_matrices, titles):
        # Plot the matrix with fixed color bar range
        im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        # Annotate diagonal values
        for i in range(min(matrix.shape[0], matrix.shape[1])):  # Ensure within bounds
            ax.text(i, i, f"{matrix[i, i]:.2f}", ha="center", va="center", color="black")

        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

        # Create space for the color bar outside the subplot
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)  # Adjust size and pad as needed
        fig.colorbar(im, cax=cax)  # Add color bar to the new axes

    plt.tight_layout()
    plt.show()

    return fig