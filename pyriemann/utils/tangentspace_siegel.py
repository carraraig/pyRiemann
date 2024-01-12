"""Tangent space for SPD/HPD matrices."""

import numpy as np

from .base_siegel import sqrtm, invsqrtm, logm, expm, tanhm, arctanhm
from .mean import mean_covariance


def _check_dimensions(X, Cref):
    n_1, n_2 = X.shape[-2:]
    n_3, n_4 = Cref.shape
    if not (n_1 == n_2 == n_3 == n_4):
        raise ValueError("Inputs have incompatible dimensions.")

def exp_map_siegel(X, Cref):
    r"""Project matrices back to the manifold by Siegel exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to Siegel manifold by Siegel Riemannian exponential map
    according to a Siegel reference matrix [1]

    Implementation based on
    https://github.com/geomstats/geomstats/blob/806764bf6dbabc0925ee0b0e54ee35ef3d0902d6/geomstats/geometry/siegel.py#L205

    [1] Cabanes, Y. (2022). Multidimensional complex stationary centered Gaussian autoregressive time series machine
        learning in Poincaré and Siegel disks: application for audio and radar clutter classification
        (Doctoral dissertation, Université de Bordeaux).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    """
    _check_dimensions(X, Cref)

    # Define Tangent vector from base point to zero (V_1)
    Identity = np.eye(Cref.shape[0], dtype=Cref.dtype)
    omega_omega_H = Cref @ Cref.conj().T
    omega_H_omega = Cref.conj().T @ Cref
    factor_1 = invsqrtm(Identity - omega_omega_H)
    factor_3 = invsqrtm(Identity - omega_H_omega)

    V_1 = factor_1 @ X @ factor_3

    # Exponential at zero (psi_1)
    if len(V_1.shape) > 2:
        prod_1_exp = V_1 @ np.transpose(V_1.conj(), axes=[0, 2, 1])
    else:
        prod_1_exp = V_1 @ V_1.conj().T
    Y = sqrtm(prod_1_exp)
    Y_inv = np.linalg.inv(Y)
    psi_1 = tanhm(Y) @ Y_inv @ V_1

    # Transport back using inverse isometry (Exp)
    factor_4_iso = sqrtm(Identity - omega_H_omega)
    factor_2_iso = psi_1 + Cref
    factor_3_iso = np.linalg.inv(Identity + (Cref.conj().T @ psi_1))

    X_new = factor_1 @ factor_2_iso @ factor_3_iso @ factor_4_iso

    return X_new

def log_map_siegel(X, Cref):
    r"""Project matrices in tangent space by Siegel logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from Siegel manifold
    to tangent space by Siegel Riemannian logarithmic map
    according to a Siegel reference matrix [1]

    [1] Cabanes, Y. (2022). Multidimensional complex stationary centered Gaussian autoregressive time series machine
        learning in Poincaré and Siegel disks: application for audio and radar clutter classification
        (Doctoral dissertation, Université de Bordeaux).

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    """
    _check_dimensions(X, Cref)

    # Define Isometry psi_1
    Identity = np.eye(Cref.shape[0])
    omega_omega_H = Cref @ Cref.conj().T
    omega_H_omega = Cref.conj().T @ Cref

    factor_1 = invsqrtm(Identity - omega_omega_H)
    factor_4 = sqrtm(Identity - omega_H_omega)
    factor_2 = X - Cref
    factor_3 = np.linalg.inv(Identity - (Cref.conj().T @ X))

    psi_1 = factor_1 @ factor_2 @ factor_3 @ factor_4

    # Define log at zero V_1
    term_1_log = psi_1 @ np.transpose(psi_1.conj(), axes=[0, 2, 1])
    X_ = sqrtm(term_1_log)
    V_1 = arctanhm(X_) @ np.linalg.inv(X_) @ psi_1

    # Logarithm to the point X_new
    X_new = sqrtm(Identity - omega_omega_H) @ V_1 @ factor_4

    return X_new

def tangent_space_siegel(X, Cref, *, metric='siegel'):
    """Transform matrices into tangent vectors.

    Transform matrices into tangent vectors, according to a reference
    matrix Cref and to a specific logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in manifold.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='siegel'
        The metric used for logarithmic map

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.

    See Also
    --------
    log_map_siegel
    upper
    """
    log_map_functions = {
        'siegel': log_map_siegel
    }
    T = log_map_functions[metric](X, Cref)

    return T


def untangent_space_siegel(T, Cref, *, metric='siegel'):
    """Transform tangent vectors back to matrices.

    Transform tangent vectors back to matrices, according to a reference
    matrix Cref and to a specific exponential map.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='siegel'
        The metric used for exponential map.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Matrices in manidold.

    See Also
    --------
    unupper
    exp_map_siegel
    """
    exp_map_functions = {
        'siegel': exp_map_siegel
    }
    X = exp_map_functions[metric](T, Cref)

    return X
