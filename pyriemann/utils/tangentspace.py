"""Tangent space for SPD/HPD matrices."""

import numpy as np

from .base import sqrtm, invsqrtm, logm, expm, tanhm, arctanhm
from .mean import mean_covariance


def _check_dimensions(X, Cref):
    n_1, n_2 = X.shape[-2:]
    n_3, n_4 = Cref.shape
    if not (n_1 == n_2 == n_3 == n_4):
        raise ValueError("Inputs have incompatible dimensions.")


def exp_map_euclid(X, Cref):
    r"""Project matrices back to manifold by Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to manifold with Euclidean exponential map
    according to a reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{X} + \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices in tangent space.
    Cref : ndarray, shape (n, m)
        The reference matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, m)
        Matrices in manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X + Cref


def exp_map_logeuclid(X, Cref):
    r"""Project matrices back to manifold by Log-Euclidean exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Log-Euclidean exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} =
        \exp(\mathbf{X} + \log(\mathbf{C}_\text{ref}))

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return expm(X + logm(Cref))


def exp_map_riemann(X, Cref, Cm12=False):
    r"""Project matrices back to manifold by Riemannian exponential map.

    The projection of a matrix :math:`\mathbf{X}` from tangent space
    to SPD/HPD manifold with Riemannian exponential map
    according to a reference SPD/HPD matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp(\mathbf{X}) \mathbf{C}_\text{ref}^{1/2}

    When Cm12=True, it returns the full Riemannian exponential map:

    .. math::
        \mathbf{X}_\text{original} = \mathbf{C}_\text{ref}^{1/2}
        \exp( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in tangent space.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.
    Cm12 : bool, default=False
        If True, it returns the full Riemannian exponential map.

    Returns
    -------
    X_original : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manifold.

    Notes
    -----
    .. versionadded:: 0.4
    """
    if Cm12:
        Cm12 = invsqrtm(Cref)
        X = Cm12 @ X @ Cm12
    C12 = sqrtm(Cref)
    return C12 @ expm(X) @ C12


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


def log_map_euclid(X, Cref):
    r"""Project matrices in tangent space by Euclidean logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from manifold
    to tangent space by Euclidean logarithmic map
    according to a reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{X} - \mathbf{C}_\text{ref}

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        Matrices in manidold.
    Cref : ndarray, shape (n, m)
        The reference matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, m)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    return X - Cref


def log_map_logeuclid(X, Cref):
    r"""Project matrices in tangent space by Log-Euclidean logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by Log-Euclidean logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log(\mathbf{X}) - \log(\mathbf{C}_\text{ref})

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD matrix.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    _check_dimensions(X, Cref)
    return logm(X) - logm(Cref)


def log_map_riemann(X, Cref, C12=False):
    r"""Project matrices in tangent space by Riemannian logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from SPD/HPD manifold
    to tangent space by Riemannian logarithmic map
    according to a SPD/HPD reference matrix :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{X}_\text{new} = \log ( \mathbf{C}_\text{ref}^{-1/2}
        \mathbf{X} \mathbf{C}_\text{ref}^{-1/2})

    When C12=True, it returns the full Riemannian logarithmic map:

    .. math::
        \mathbf{X}_\text{new} = \mathbf{C}_\text{ref}^{1/2}
        \log( \mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
        \mathbf{C}_\text{ref}^{-1/2}) \mathbf{C}_\text{ref}^{1/2}

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in SPD/HPD manidold.
    Cref : ndarray, shape (n, n)
        The reference SPD/HPD matrix.
    C12 : bool, default=False
        If True, it returns the full Riemannian logarithmic map.

    Returns
    -------
    X_new : ndarray, shape (..., n, n)
        Matrices projected in tangent space.

    Notes
    -----
    .. versionadded:: 0.4
    """
    _check_dimensions(X, Cref)
    Cm12 = invsqrtm(Cref)
    X_new = logm(Cm12 @ X @ Cm12)
    if C12:
        C12 = sqrtm(Cref)
        X_new = C12 @ X_new @ C12
    return X_new


def log_map_siegel(X, Cref):
    r"""Project matrices in tangent space by Siegel logarithmic map.

    The projection of a matrix :math:`\mathbf{X}` from Siegel manifold
    to tangent space by Siegel Riemannian logarithmic map
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


def upper(X):
    r"""Return the weighted upper triangular part of matrices.

    This function computes the minimal representation of a matrix in tangent
    space [1]_: it keeps the upper triangular part of the symmetric/Hermitian
    matrix and vectorizes it by applying unity weight for diagonal elements and
    :math:`\sqrt{2}` weight for out-of-diagonal elements.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices.

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Weighted upper triangular parts of symmetric/Hermitian matrices.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Pedestrian detection via classification on Riemannian manifolds
        <https://ieeexplore.ieee.org/document/4479482>`_
        O. Tuzel, F. Porikli, and P. Meer. IEEE Transactions on Pattern
        Analysis and Machine Intelligence, Volume 30, Issue 10, October 2008.
    """
    n = X.shape[-1]
    if X.shape[-2] != n:
        raise ValueError("Matrices must be square")
    idx = np.triu_indices_from(np.empty((n, n)))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n, n)), 1) + np.eye(n))[idx]
    T = coeffs * X[..., idx[0], idx[1]]
    return T


def unupper(T):
    """Inverse upper function.

    This function is the inverse of upper function: it reconstructs symmetric/
    Hermitian matrices from their weighted upper triangular parts.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Weighted upper triangular parts of symmetric/Hermitian matrices.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Symmetric/Hermitian matrices.

    See Also
    --------
    upper

    Notes
    -----
    .. versionadded:: 0.4
    """
    dims = T.shape
    n = int((np.sqrt(1 + 8 * dims[-1]) - 1) / 2)
    X = np.empty((*dims[:-1], n, n), dtype=T.dtype)
    idx = np.triu_indices_from(np.empty((n, n)))
    X[..., idx[0], idx[1]] = T
    idx = np.triu_indices_from(np.empty((n, n)), k=1)
    X[..., idx[0], idx[1]] /= np.sqrt(2)
    X[..., idx[1], idx[0]] = X[..., idx[0], idx[1]].conj()
    return X


def tangent_space(X, Cref, *, metric='riemann'):
    """Transform matrices into tangent vectors.

    Transform matrices into tangent vectors, according to a reference
    matrix Cref and to a specific logarithmic map.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Matrices in manidold.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='riemann'
        The metric used for logarithmic map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.

    See Also
    --------
    log_map_euclid
    log_map_logeuclid
    log_map_riemann
    log_map_siegel
    upper
    """
    log_map_functions = {
        'euclid': log_map_euclid,
        'logeuclid': log_map_logeuclid,
        'riemann': log_map_riemann,
        'siegel': log_map_siegel
    }
    X_ = log_map_functions[metric](X, Cref)
    T = upper(X_)

    return T


def untangent_space(T, Cref, *, metric='riemann'):
    """Transform tangent vectors back to matrices.

    Transform tangent vectors back to matrices, according to a reference
    matrix Cref and to a specific exponential map.

    Parameters
    ----------
    T : ndarray, shape (..., n * (n + 1) / 2)
        Tangent vectors.
    Cref : ndarray, shape (n, n)
        The reference matrix.
    metric : string, default='riemann'
        The metric used for exponential map, can be: 'euclid', 'logeuclid',
        'riemann'.

    Returns
    -------
    X : ndarray, shape (..., n, n)
        Matrices in manidold.

    See Also
    --------
    unupper
    exp_map_euclid
    exp_map_logeuclid
    exp_map_riemann
    exp_map_siegel
    """
    X_ = unupper(T)
    exp_map_functions = {
        'euclid': exp_map_euclid,
        'logeuclid': exp_map_logeuclid,
        'riemann': exp_map_riemann,
        'siegel': exp_map_siegel
    }
    X = exp_map_functions[metric](X_, Cref)

    return X


###############################################################################


# NOT IN API
def transport(Covs, Cref, metric='riemann'):
    """Parallel transport of a set of SPD matrices towards a reference matrix.

    Parameters
    ----------
    Covs : ndarray, shape (n_matrices, n, n)
        Set of SPD matrices.
    Cref : ndarray, shape (n, n)
        The reference SPD matrix.
    metric : string, default='riemann'
        The metric used for mean, can be: 'euclid', 'logeuclid', 'riemann'.

    Returns
    -------
    out : ndarray, shape (n_matrices, n, n)
        Set of transported SPD matrices.
    """
    C = mean_covariance(Covs, metric=metric)
    iC = invsqrtm(C)
    E = sqrtm(iC @ Cref @ iC)
    out = E @ Covs @ E.T
    return out
