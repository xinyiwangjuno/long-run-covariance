import numpy as np

# Helper: Brownian motion with 0 prepended
def brown_vec(refinement):
    dt = 1.0 / (refinement - 1)
    steps = np.random.normal(0, np.sqrt(dt), size=refinement - 1)
    return np.concatenate(([0], np.cumsum(steps)))

# Brownian Motion Generator
def brown_mat(N, refinement):
    mat = np.zeros((refinement, N))
    for c in range(N):
        vec = np.cumsum(np.random.normal(0, 1, size=refinement - 1))
        mat[1:, c] = vec
    return mat

# MA(0.5, 1)
def bm_lag_mat(N, refinement):
    firstmat = brown_mat(N + 1, refinement)
    mat = 0.5 * firstmat[:, :-1] + 0.5 * firstmat[:, 1:]
    return mat

# MA(0.5, 4)
def ma5_mat(N, refinement, theta=0.5):
    firstmat = brown_mat(N + 4, refinement)
    mat = sum(theta * firstmat[:, i:i+N] for i in range(5))
    return mat

# MA_phi kernel: min(i,j)/ref
def ma_phi(ref, k):
    i = np.arange(1, ref + 1).reshape(-1, 1)
    j = np.arange(1, ref + 1).reshape(1, -1)
    return k * np.minimum(i, j) / ref

# Kernel-based integral
def fun_integral(ref, Mat, X):
    return Mat @ X / ref

# MA_phi_4 functional process
def fun_ma_mat(N, refinement):
    Mat = ma_phi(refinement, 1.5)
    first_mat = brown_mat(N + 4, refinement)
    final_mat = np.zeros((refinement, N))
    for c in range(N):
        vec = (
            fun_integral(refinement, Mat, first_mat[:, c]) +
            fun_integral(refinement, Mat, first_mat[:, c + 1]) +
            fun_integral(refinement, Mat, first_mat[:, c + 2]) +
            fun_integral(refinement, Mat, first_mat[:, c + 3]) +
            first_mat[:, c + 4]
        )
        final_mat[:, c] = vec
    return final_mat

# MA(0.5, 8)
def ma9_mat(N, refinement, theta=0.5):
    first_mat = brown_mat(N + 8, refinement)
    return sum(theta * first_mat[:, i:i + N] for i in range(9))

# FAR(0.5, 1)
def zlag_mat_05(N, refinement):
    first_mat = brown_mat(N + 50, refinement)
    mat = np.zeros((refinement, N + 50))
    mat[:, 0] = first_mat[:, 0]
    for c in range(1, N + 50):
        mat[:, c] = first_mat[:, c] + 0.5 * mat[:, c - 1]
    return mat[:, 50:]

# FAR_psi_1 kernel
def fun_kernel(ref, k):
    i = np.arange(1, ref + 1).reshape(-1, 1)
    j = np.arange(1, ref + 1).reshape(1, -1)
    return k * np.exp(0.5 * ((i / ref) ** 2 + (j / ref) ** 2))

# FAR_psi_1 process
def fun_ar_mat(N, refinement):
    Mat = fun_kernel(refinement, 0.34)
    first_mat = np.zeros((refinement, N + 50))
    first_mat[:, 0] = brown_vec(refinement)
    for i in range(1, N + 50):
        first_mat[:, i] = fun_integral(refinement, Mat, first_mat[:, i - 1]) + brown_vec(refinement)
    return first_mat[:, 50:]


def dgp_fun(seed, N, refinement, dgp):
    # seed: seed number
    # N: number of curves
    # refinement: number of points (discretization steps) per curve
    # dgp: choose among 7 DGPs
    np.random.seed(122 + seed)

    if dgp == "MA_1_0":
        data1 = brown_mat(N, refinement)
    elif dgp == "MA_0.5_1":
        data1 = bm_lag_mat(N, refinement)
    elif dgp == "MA_0.5_4":
        data1 = ma5_mat(N, refinement)  # make sure theta is set inside or passed in
    elif dgp == "MA_psi_4":
        data1 = fun_ma_mat(N, refinement)
    elif dgp == "MA_0.5_8":
        data1 = ma9_mat(N, refinement)  # again, set theta as needed
    elif dgp == "FAR_0.5_1":
        data1 = zlag_mat_05(N, refinement)
    elif dgp == "FAR_psi_1":
        data1 = fun_ar_mat(N, refinement)
    else:
        raise ValueError(f"Unknown DGP type: {dgp}")

    return data1