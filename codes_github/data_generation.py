import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def generate_Y_case1(d, theta, J, n, plot=False):
    """
    Generates X_t and Y with integral transformations for each curve instance.

    Parameters:
        d (int): Number of curves (or instances of X_t)
        theta (float): Parameter for FMA(1)
        J (int): Number of functional coefficients
        n (int): Number of discrete points for each curve (representation of [0,1])
        plot (bool): If True, plots X_t and Y

    Returns:
        X (torch.Tensor): Tensor of shape (n, d) representing the X_t values
        Y (torch.Tensor): Tensor of shape (1, d) representing the Y values
    """
    # Discretized time for integration
    time_grid = np.linspace(0, 1, n)

    # Step 1: Generate X_t as an FMA(1) process with each X_t as a discrete representation of a curve
    np.random.seed(42)
    Z = np.random.normal(0, 1, (n + 1, d))  # Generate Gaussian noise for each discrete point
    X_np = np.zeros((n, d))
    X_np[0] = Z[0]
    for t in range(1, n):
        X_np[t] = theta * Z[t-1] + Z[t]  # Apply FMA(1) process for each time step across all curves

    # Step 2: Define the functional coefficients beta_j(t) as functions over [0, 1]
    betas_np = [np.sin((j + 1) * np.pi * time_grid) for j in range(J)]  # List of J beta functions

    # Step 3: Compute Y as integrals over the product of beta_j(t) and each X[:, t] over [0,1]
    Y_np = np.zeros(d)
    for curve in range(d):
        integrals = []
        for j in range(J):
            # Integral approximation via Riemann sum over the discrete time points for each curve
            integral = np.trapz(betas_np[j] * X_np[:, curve], time_grid)
            integrals.append(integral)
        Y_np[curve] = np.sum(integrals) + np.random.normal(0, 1)  # Add Gaussian noise epsilon

    # Convert X and Y to PyTorch tensors
    #X = torch.tensor(X_np, dtype=torch.float32)
    #Y = torch.tensor(Y_np, dtype=torch.float32).view(1, d)  # Reshape Y to (1, d)

    # Plot if plot=True
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot all X_t curves
        for i in range(d):
            axes[0].plot(time_grid, X_np[:, i], label=f'Curve {i+1}')
        axes[0].set_title('X_t Curves')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X_t')

        # Plot Y values
        axes[1].plot(range(d), Y_np, 'o-', color='tab:orange')
        axes[1].set_title('Y values')
        axes[1].set_xlabel('Curve index')
        axes[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

    return X_np, Y_np

def generate_Y_case2(d, theta, J, n,lambda_param, alpha, sigma, plot=False):
    """
    Generates Y with nonlinear transformations and interaction terms for each curve instance.

    Parameters:
        d (int): Number of curves (or instances of X_t)
        theta (float): Parameter for FMA(1)
        J (int): Number of functional coefficients
        n (int): Number of discrete points for each curve (representation of [0,1])
        lambda_param (float): Weight for the interaction terms
        alpha (float): Weight for the exponential term
        sigma (float): Standard deviation of Gaussian noise
        plot (bool): If True, plots X_t and Y

    Returns:
        X (torch.Tensor): Tensor of shape (n, d) representing the X_t values
        Y (torch.Tensor): Tensor of shape (1, d) representing the Y values
    """
    time_grid = np.linspace(0, 1, n)

    # Step 1: Generate X_t as an FMA(1) process with each X_t as a discrete representation of a curve
    np.random.seed(42)
    Z = np.random.normal(0, 1, (n + 1, d))  # Generate Gaussian noise for each discrete point
    X_np = np.zeros((n, d))
    X_np[0] = Z[0]
    for t in range(1, n):
        X_np[t] = theta * Z[t - 1] + Z[t]  # Apply FMA(1) process for each time step across all curves

    # Step 2: Define the functional coefficients beta_j(t) as functions over [0, 1]
    betas_np = [np.sin((j + 1) * np.pi * time_grid) for j in range(J)]  # List of J beta functions

    # Step 3: Compute Y with nonlinear transformations, interaction terms, and exponential term
    Y_np = np.zeros(d)
    for curve in range(d):
        integrals = []
        for j in range(J):
            # Integral approximation via Riemann sum over the discrete time points with nonlinear sin transformation
            integral = np.trapz(np.sin(betas_np[j] * X_np[:, curve]), time_grid)
            integrals.append(integral)

        # Nonlinear interaction terms with cos transformation
        interaction_sum = 0
        for t1 in range(n):
            for t2 in range(t1 + 1, n):  # Only consider pairs t1 < t2 for unique interactions
                interaction_sum += np.cos(X_np[t1, curve] * X_np[t2, curve])

        # Nonlinear exponential term based on average squared values
        exp_term = np.exp(np.mean(X_np[:, curve] ** 2))

        # Combine terms with weights and Gaussian noise
        Y_np[curve] = np.sum(integrals) + lambda_param * interaction_sum + alpha * exp_term + np.random.normal(0, sigma)

    # Convert X and Y to PyTorch tensors
    #X = torch.tensor(X_np, dtype=torch.float32)
    #Y = torch.tensor(Y_np, dtype=torch.float32).view(1, d)  # Reshape Y to (1, d)

    # Plot if plot = True
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot all X_t curves
        for i in range(d):
            axes[0].plot(time_grid, X_np[:, i], label=f'Curve {i + 1}')
        axes[0].set_title('X_t Curves')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X_t')

        # Plot Y values
        axes[1].plot(range(d), Y_np, 'o-', color='tab:orange')
        axes[1].set_title('Y values')
        axes[1].set_xlabel('Curve index')
        axes[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

    return X_np, Y_np

# Define deterministic kernel psi(t, s)
def psi(t, s):
    return np.exp(-np.abs(t - s))  # Example: exponential decay kernel

# Define random kernel operator phi_n(t, s, epsilon_n(u))
def phi(t, s, epsilon):
    """Random kernel function based on noise."""
    return np.sin(t * s + epsilon)  # Example: sine-based interaction with noise


def generate_Y_case3(d, n, J,lambda_param, alpha, sigma, plot=False):
    """
    Generates Y with nonlinear transformations and interaction terms for each curve instance.

    Parameters:
        d (int): Number of curves (or instances of X_t)
        n (int): Number of discrete points for each curve (representation of [0,1])
        plot (bool): If True, plots X_t and Y
        J (int): Number of functional coefficients
        lambda_param (float): Weight for the interaction terms
        alpha (float): Weight for the exponential term
        sigma (float): Standard deviation of Gaussian noise

    Returns:
        X (torch.Tensor): Tensor of shape (n, d) representing the X_t values
        Y (torch.Tensor): Tensor of shape (1, d) representing the Y values
    """
    np.random.seed(42)
    # Initialize the process
    X_np = np.zeros((n,d))
    epsilon = np.random.randn(n, d)  # Random noise for each time point and curve
    T = 1
    time_grid = np.linspace(0, T, n)

    # Functional bilinear process
    for t_idx, t in enumerate(time_grid): 
        for curve in range(d):  
            integral_psi = np.sum([psi(t, s) * X_np[s_idx, curve] for s_idx, s in enumerate(time_grid)]) * (T / n)
        
            integral_phi = np.sum([
                phi(t, s, epsilon[u_idx, curve]) * X_np[s_idx, curve]
                for s_idx, s in enumerate(time_grid)
                for u_idx in range(n)
            ]) * (T / n)**2
        
            # Update process
            X_np[t_idx, curve] = integral_psi + integral_phi + epsilon[t_idx, curve]


    # Step 2: Define the functional coefficients beta_j(t) as functions over [0, 1]
    betas_np = [np.sin((j + 1) * np.pi * time_grid) for j in range(J)]  

    # Step 3: Compute Y with nonlinear transformations, interaction terms, and exponential term
    Y_np = np.zeros(d)
    for curve in range(d):
        integrals = []
        for j in range(J):
            # Integral approximation via Riemann sum over the discrete time points with nonlinear sin transformation
            integral = np.trapz(np.sin(betas_np[j] * X_np[:, curve]), time_grid)
            integrals.append(integral)

        # Nonlinear interaction terms with cos transformation
        interaction_sum = 0
        for t1 in range(n):
            for t2 in range(t1 + 1, n):  # Only consider pairs t1 < t2 for unique interactions
                interaction_sum += np.cos(X_np[t1, curve] * X_np[t2, curve])

        # Nonlinear exponential term based on average squared values
        exp_term = np.exp(np.mean(X_np[:, curve] ** 2))

        # Combine terms with weights and Gaussian noise
        Y_np[curve] = np.sum(integrals) + lambda_param * interaction_sum + alpha * exp_term + np.random.normal(0, sigma)

    # Convert X and Y to PyTorch tensors
    #X = torch.tensor(X_np, dtype=torch.float32)
    #Y = torch.tensor(Y_np, dtype=torch.float32).view(1, d)  # Reshape Y to (1, d)

    # Plot if plot = True
    if plot:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

        # Plot all X_t curves
        for i in range(d):
            axes[0].plot(time_grid, X_np[:, i], label=f'Curve {i + 1}')
        axes[0].set_title('X_t Curves')
        axes[0].set_xlabel('Time')
        axes[0].set_ylabel('X_t')

        # Plot Y values
        axes[1].plot(range(d), Y_np, 'o-', color='tab:orange')
        axes[1].set_title('Y values')
        axes[1].set_xlabel('Curve index')
        axes[1].set_ylabel('Y')

        plt.tight_layout()
        plt.show()

    return X_np, Y_np




    
    







