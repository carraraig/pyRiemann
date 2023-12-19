"""Means of matrices that belong to Siegel Disk"""

from copy import deepcopy
import numpy as np
import warnings

from .ajd import ajd_pham
from .base_siegel import sqrtm, invsqrtm, logm, expm, powm, arctanhm, tanhm
from .distance import distance_riemann
from .geodesic import geodesic_riemann
from .utils import check_weights


def _deprecate_covmats(covmats, X):
    if covmats is not None:
        print("DeprecationWarning: input covmats has been renamed into X and "
              "will be removed in 0.8.0.")
        X = covmats
    return X


def mean_euclid(X=None, sample_weight=None, covmats=None):
    r"""Mean of matrices according to the Euclidean metric.

    .. math::
        \mathbf{M} = \sum_i w_i \ \mathbf{X}_i

    This mean is also called arithmetic.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, m)
        Set of matrices.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.

    Returns
    -------
    M : ndarray, shape (n, m)
        Euclidean mean.

    See Also
    --------
    mean_covariance
    """
    X = _deprecate_covmats(covmats, X)
    return np.average(X, axis=0, weights=sample_weight)

def mean_siegel(X=None, tol=10e-9, maxiter=50, init=None, sample_weight=None):
    r"""Mean of siegel matrices [2] according to the Siegel metric [1].

    [1] Cabanes, Y. (2022). Multidimensional complex stationary centered Gaussian autoregressive time series machine
        learning in Poincaré and Siegel disks: application for audio and radar clutter classification
        (Doctoral dissertation, Université de Bordeaux).

    [2] Jeuris, B., & Vandebril, R. (2016). The Kahler mean of block-Toeplitz matrices with Toeplitz structured blocks.
        SIAM Journal on matrix analysis and applications, 37(3), 1151-1175.

    Parameters
    ----------
    covmats : ndarray, shape (n_matrices, n, n)
        Set of SPD/HPD matrices.
    tol : float, default=10e-9
        The tolerance to stop the gradient descent.
    maxiter : int, default=50
        The maximum number of iterations.
    init : None | ndarray, shape (n, n), default=None
        A SPD/HPD matrix used to initialize the gradient descent.
        If None, the weighted Euclidean mean is used.

    Returns
    -------
    C : ndarray, shape (n, n)
        Siegel mean.

    See Also
    --------
    mean_covariance
    """
    n_matrices, _, _ = X.shape
    sample_weight = check_weights(sample_weight, n_matrices)
    if init is None:
        C = mean_euclid(X, sample_weight=sample_weight)
    else:
        C = init

    nu = 1.0
    tau = np.finfo(np.float64).max
    crit = np.finfo(np.float64).max
    for _ in range(maxiter):
        Cov_Tang = log_map_siegel(X, C)
        J = mean_euclid(Cov_Tang)
        C = exp_map_siegel(nu * J, C)

        crit = np.linalg.norm(J, ord='fro')
        h = nu * crit
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu
        if crit <= tol or nu <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    return C

###############################################################################


mean_functions = {
    'siegel': mean_siegel
}


def _check_mean_function(metric):
    """Check mean function."""
    if isinstance(metric, str):
        if metric not in mean_functions.keys():
            raise ValueError(f"Unknown mean metric '{metric}'")
        else:
            metric = mean_functions[metric]
    elif not hasattr(metric, '__call__'):
        raise ValueError("Mean metric must be a function or a string "
                         f"(Got {type(metric)}.")
    return metric


def mean_covariance_siegel(X=None, metric='siegel', sample_weight=None, covmats=None,
                    **kwargs):
    """Mean of matrices according to a metric.

    Compute the mean of a set of matrices according to a metric [1]_.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n, n)
        Set of matrices.
    metric : string, default='siegel'
        The metric for mean, can be: 'siegel' or a callable function.
    sample_weight : None | ndarray, shape (n_matrices,), default=None
        Weights for each matrix. If None, it uses equal weights.
    **kwargs : dict
        The keyword arguments passed to the sub function.

    Returns
    -------
    M : ndarray, shape (n, n)
        Mean of matrices.

    References
    ----------
    .. [1] `Review of Riemannian distances and divergences, applied to
        SSVEP-based BCI
        <https://hal.archives-ouvertes.fr/LISV/hal-03015762v1>`_
        S. Chevallier, E. K. Kalunga, Q. Barthélemy, E. Monacelli.
        Neuroinformatics, Springer, 2021, 19 (1), pp.93-106
    """
    X = _deprecate_covmats(covmats, X)
    mean_function = _check_mean_function(metric)
    M = mean_function(
        X,
        sample_weight=sample_weight,
        **kwargs,
    )
    return M


###############################################################################
def _check_dimensions(X, Cref):
    n_1, n_2 = X.shape[-2:]
    n_3, n_4 = Cref.shape
    if not (n_1 == n_2 == n_3 == n_4):
        raise ValueError("Inputs have incompatible dimensions.")
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