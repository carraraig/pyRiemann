"""Distances between SPD/HPD matrices."""

import numpy as np
from scipy.linalg import eigvalsh, solve
from sklearn.metrics import euclidean_distances
import scipy

from .base_siegel import logm, sqrtm, invsqrtm


def _check_inputs(A, B):
    if not isinstance(A, np.ndarray) or not isinstance(B, np.ndarray):
        raise ValueError("Inputs must be ndarrays")
    if not A.shape == B.shape:
        raise ValueError("Inputs must have equal dimensions")
    if A.ndim < 2:
        raise ValueError("Inputs must be at least a 2D ndarray")

def distance_siegel(A, B):
    r"""Distance in Siegel Space

    The distance between two matrices belong to Siegel Space

    $\begin{aligned} & d_{\mathbb{S D}_N}^2(\Omega, \Psi)=
     =\frac{1}{4} \operatorname{trace}\left(\log ^2\left(\frac{I+C^{1 / 2}}{I-C^{1 / 2}}\right)\right) \\ &
     =\operatorname{trace}\left(\operatorname{arctanh}^2\left(C^{1 / 2}\right)\right) \\ &
     \text { with } C=(\Psi-\Omega)\left(I-\Omega^H \Psi\right)^{-1}\left(\Psi^H-\Omega^H\right)\left(I-\Omega \Psi^H\right)^{-1} \text {. } \\ & \end{aligned}$

    Implementation based on
    https://github.com/geomstats/geomstats/blob/806764bf6dbabc0925ee0b0e54ee35ef3d0902d6/geomstats/geometry/siegel.py#L205

    Parameters
    ----------
    A : ndarray, shape (..., n, m)
        First matrices, at least 2D ndarray.
    B : ndarray, shape (..., n, m)
        Second matrices, same dimensions as A.

    Returns
    -------
    d : float or ndarray, shape (...,)
        Siegel distance between A and B.

    See Also
    --------
    distance
    """
    _check_inputs(A, B)
    if len(A.shape) > 2:
        A_conjT = np.transpose(A.conj(), axes=[0, 2, 1])
    else:
        A_conjT = A.conj().T

    if len(B.shape) > 2:
        B_conjT = np.transpose(B.conj(), axes=[0, 2, 1])
    else:
        B_conjT = B.conj().T

    term1 = B - A
    term2 = np.linalg.inv(np.eye(A.shape[0]) - A_conjT @ B)
    term3 = B_conjT - A_conjT
    term4 = np.linalg.inv(np.eye(A.shape[0]) - A @ B_conjT)
    C = term1 @ term2 @ term3 @ term4

    prod_power_one_half = sqrtm(C)
    num = np.eye(A.shape[0]) + prod_power_one_half
    den = scipy.linalg.inv(np.eye(A.shape[0]) - prod_power_one_half)

    frac = num @ den

    logarithm = logm(frac)
    sq_dist = 0.25 * np.trace(logarithm @ logarithm)
    sq_dist = np.real(sq_dist)

    return np.maximum(sq_dist, 0)


distance_functions = {
    'siegel': distance_siegel,
}


def _check_distance_function(metric):
    """Check distance function."""
    if isinstance(metric, str):
        if metric not in distance_functions.keys():
            raise ValueError(f"Unknown distance metric '{metric}'")
        else:
            metric = distance_functions[metric]
    elif not hasattr(metric, '__call__'):
        raise ValueError("Distance metric must be a function or a string "
                         f"(Got {type(metric)}.")
    return metric


def distance_siegel(A, B, metric='siegel', squared=False):
    """Distance between matrices according to a metric.

    Compute the distance between two matrices A and B according to a metric
    [1]_, or between a set of matrices A and another matrix B.

    Parameters
    ----------
    A : ndarray, shape (n, n) or shape (n_matrices, n, n)
        First matrix, or set of matrices.
    B : ndarray, shape (n, n)
        Second matrix.
    metric : string, default='siegel'
        The metric for distance, can be: 'siegel' or a callable function.
    squared : bool, default False
        Return squared distance.

    Returns
    -------
    d : float or ndarray, shape (n_matrices, 1)
        Distance between A and B.

    References
    ----------
    .. [1] Cabanes, Y. (2022). Multidimensional complex stationary centered Gaussian autoregressive time series machine
        learning in Poincaré and Siegel disks: application for audio and radar clutter classification
        (Doctoral dissertation, Université de Bordeaux).
    """
    distance_function = _check_distance_function(metric)

    shape_A, shape_B = A.shape, B.shape
    if shape_A == shape_B:
        d = distance_function(A, B)
    elif len(shape_A) == 3 and len(shape_B) == 2:
        d = np.empty((shape_A[0], 1))
        for i in range(shape_A[0]):
            d[i] = distance_function(A[i], B)
    else:
        raise ValueError("Inputs have incompatible dimensions.")

    return d