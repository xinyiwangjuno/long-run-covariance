import numpy as np
from skfda.representation.basis import (
    FourierBasis,
    BSplineBasis)
import skfda

## MA(1) model

def generate_fma1_coef(d, J, Sigma=None, Theta=None, seed=None):
    """
    Generate MA(1)-like coefficient matrix for coef @ basis representation.

    Args:
        d (int): Number of curves (samples).
        J (int): Number of basis functions (features).
        Sigma (np.ndarray or None): Optional vector of std deviations (length J).
        Theta (np.ndarray or None): Optional J x J operator matrix.
        seed (int or None): Random seed.

    Returns:
        np.ndarray: Coefficient matrix of shape (d, J).
    """

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Default Theta and Sigma
    if Theta is None:
        Theta = 0.8 * np.eye(J)
    if Sigma is None:
        Sigma = 1.0 / np.arange(1, J + 1)
    
    Fbasis = FourierBasis(domain_range=(0, 1), n_basis=J)
    G = Fbasis.gram_matrix()
    norms = np.sqrt(np.diag(G))
    print("norms of the basis are:",norms)
    #Fbasis.plot()
    #plt.show()

    # Step 1: Generate Z lag-0 matrix (shape d x J)
    zlag0 = np.random.normal(loc=0.0, scale=Sigma[None, :], size=(d, J))  # broadcast per-column

    # Step 2: Generate Z lag-1 matrix via Theta (shape d x J)
    zlag1 = np.zeros((d, J))
    for i in range(1, d):
        zlag1[i, :] = zlag0[i - 1, :] @ Theta.T  # Note: @ Theta.T

    # Final coefficient matrix (d x J)
    coef = zlag0 + zlag1
    coef -= coef.mean(axis=0, keepdims=True)

    fd_basis= skfda.FDataBasis(
    basis=Fbasis,
    coefficients= coef)
    return coef,Fbasis,fd_basis


## MA(2) model

def generate_fma2_coef(d, J, Sigma=None, Theta1=None, Theta2=None, seed=None):
    """
    Generate FMA(2)-like coefficient matrix for coef @ basis representation.

    Args:
        d (int): Number of curves (samples).
        J (int): Number of basis functions (features).
        Sigma (np.ndarray or None): Optional vector of std deviations (length J).
        Theta1 (np.ndarray or None): Optional J x J operator matrix for lag-1.
        Theta2 (np.ndarray or None): Optional J x J operator matrix for lag-2.
        seed (int or None): Random seed.

    Returns:
        tuple: (coef matrix, basis object, fd object)
    """
    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Defaults
    if Theta1 is None:
        Theta1 = 0.8 * np.eye(J)
    if Theta2 is None:
        Theta2 = 0.8 * np.eye(J)
    if Sigma is None:
        Sigma = 1.0 / np.arange(1, J + 1)

    # Create Fourier basis
    Fbasis = FourierBasis(domain_range=(0, 1), n_basis=J)
    G = Fbasis.gram_matrix()
    norms = np.sqrt(np.diag(G))
    print("norms of the basis are:", norms)

    # Step 1: Generate Z lag-0 (shape: d x J)
    zlag0 = np.random.normal(loc=0.0, scale=Sigma[None, :], size=(d, J))

    # Step 2: Generate Z lag-1 (Theta1 * zlag0[t-1])
    zlag1 = np.zeros((d, J))
    for t in range(1, d):
        zlag1[t, :] = zlag0[t - 1, :] @ Theta1.T

    # Step 3: Generate Z lag-2 (Theta2 * zlag0[t-2])
    zlag2 = np.zeros((d, J))
    for t in range(2, d):
        zlag2[t, :] = zlag0[t - 2, :] @ Theta2.T

    # Final coefficient matrix
    coef = zlag0 + zlag1 + zlag2
    coef -= coef.mean(axis=0, keepdims=True)


    fd_basis = skfda.FDataBasis(
        basis=Fbasis,
        coefficients=coef
    )

    return coef, Fbasis, fd_basis


def generate_far2_coef(d, J, Sigma=None, Psi1=None, Psi2=None, seed=None):
    """
    Generate FAR(2)-like coefficient matrix for coef @ Fourier basis representation.

    Args:
        d (int): Number of curves (samples).
        J (int): Number of Fourier basis functions (features).
        Sigma (np.ndarray or None): Optional vector of std deviations (length J).
        Psi1 (np.ndarray or None): Optional J x J operator matrix for lag-1.
        Psi2 (np.ndarray or None): Optional J x J operator matrix for lag-2.
        seed (int or None): Random seed.

    Returns:
        tuple: (coef matrix of shape (d, J), FourierBasis object, FDataBasis object)
    """

    # Set random seed
    if seed is not None:
        np.random.seed(seed)

    # Default Psi1 and Psi2
    if Psi1 is None:
        Psi1 = np.zeros((J, J))
    if Psi2 is None:
        Psi2 = np.zeros((J, J))
    if Sigma is None:
        Sigma = 1.0 / np.arange(1, J + 1)

    # Create Fourier basis
    Fbasis = FourierBasis(domain_range=(0, 1), n_basis=J)
    G = Fbasis.gram_matrix()
    norms = np.sqrt(np.diag(G))
    print("norms of the basis are:", norms)

    # Initialize coefficient matrix
    coef = np.zeros((d, J))

    # First two curves are white noise
    coef[0, :] = np.random.normal(loc=0.0, scale=Sigma)
    coef[1, :] = np.random.normal(loc=0.0, scale=Sigma)

    # Recursively generate FAR(2) coefficients
    for t in range(2, d):
        noise = np.random.normal(loc=0.0, scale=Sigma)
        coef[t, :] = coef[t - 1, :] @ Psi1.T + coef[t - 2, :] @ Psi2.T + noise

    
    coef -= coef.mean(axis=0, keepdims=True)
    # Wrap into FDataBasis
    fd_basis = skfda.FDataBasis(
        basis=Fbasis,
        coefficients=coef
    )

    return coef, Fbasis, fd_basis


# ARMA(2,2)
def generate_farma22_coef(d, J, Phi1=None, Phi2=None, Theta1=None, Theta2=None, Sigma=None, seed=None):
    """
    Simulate a Functional ARMA(2,2) process using Fourier basis.

    Args:
        d (int): Number of curves (samples).
        J (int): Number of Fourier basis functions.
        Phi1 (np.ndarray): J x J AR(1) operator.
        Phi2 (np.ndarray): J x J AR(2) operator.
        Theta1 (np.ndarray): J x J MA(1) operator.
        Theta2 (np.ndarray): J x J MA(2) operator.
        Sigma (np.ndarray): Optional vector of std deviations (length J).
        seed (int or None): Random seed.

    Returns:
        tuple: (coef matrix of shape (d, J), FourierBasis object, FDataBasis object)
    """

    if seed is not None:
        np.random.seed(seed)

    # Default operators
    if Phi1 is None:
        Phi1 = 0.8 * np.eye(J)
    if Phi2 is None:
        Phi2 = np.zeros((J, J))
    if Theta1 is None:
        Theta1 = 0.8 * np.eye(J)
    if Theta2 is None:
        Theta2 = 0.8 * np.eye(J)
    if Sigma is None:
        Sigma = 1.0 / np.arange(1, J + 1)

    # Create Fourier basis
    Fbasis = FourierBasis(domain_range=(0, 1), n_basis=J)
    G = Fbasis.gram_matrix()
    norms = np.sqrt(np.diag(G))
    print("norms of the basis are:", norms)

    # Initialize white noise and coefficient matrices
    noise = np.random.normal(loc=0.0, scale=Sigma[None, :], size=(d, J))  # shape (d, J)
    coef = np.copy(noise)

    # Recursion: ARMA(2,2)
    for t in range(2, d):
        coef[t, :] = (
            coef[t - 1, :] @ Phi1.T +
            coef[t - 2, :] @ Phi2.T +
            noise[t, :] +
            noise[t - 1, :] @ Theta1.T +
            noise[t - 2, :] @ Theta2.T
        )
    
    coef -= coef.mean(axis=0, keepdims=True)

    # Wrap in FDataBasis
    fd_basis = skfda.FDataBasis(
        basis=Fbasis,
        coefficients=coef
    )

    return coef, Fbasis, fd_basis
