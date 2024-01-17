"""Tangent space functions."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.utils import check_version
from .utils.mean import mean_covariance
from pyriemann.utils.mean_siegel import mean_covariance_siegel
from .utils.tangentspace import tangent_space, untangent_space, unupper
from pyriemann.utils.tangentspace_siegel import tangent_space_siegel

if check_version('sklearn', '0.17'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
else:
    from sklearn.lda import LDA


class TangentSpace(BaseEstimator, TransformerMixin):

    """Tangent space project TransformerMixin.

    Tangent space projection map a set of SPD matrices to their
    tangent space according to [1]_. The Tangent space projection can be
    seen as a kernel operation, cf [2]_. After projection, each matrix is
    represented as a vector of size :math:`n (n+1)/2`, where :math:`n` is the
    dimension of the SPD matrices.

    Tangent space projection is useful to convert SPD matrices in
    Euclidean vectors while conserving the inner structure of the manifold.
    After projection, standard processing and vector-based classification can
    be applied.

    Tangent space projection is a local approximation of the manifold. it takes
    one parameter, the reference point, that is usually estimated using the
    geometric mean of the SPD matrices set you project. If the function
    `fit` is not called, the identity matrix will be used as reference point.
    This can lead to serious degradation of performances.
    The approximation will be bigger if the matrices in the set are scattered
    in the manifold, and lower if they are grouped in a small region of the
    manifold.

    After projection, it is possible to go back in the manifold using the
    inverse transform.

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric) and for tangent
        space map (see `tangent_space` for the list of supported metric).
        The metric could be a dict with two keys, `mean` and `map` in
        order to pass different metrics for the reference matrix estimation
        and the tangent space mapping.
    tsupdate : bool, default=False
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.

    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.

    See Also
    --------
    FgMDM
    FGDA

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)

        self.reference_ = mean_covariance(
            X,
            metric=self.metric_mean,
            sample_weight=sample_weight
        )
        return self

    def _check_metric(self, metric):

        if isinstance(metric, str):
            metric_mean = metric
            metric_map = metric

        elif isinstance(metric, dict):
            # check keys
            for key in ['mean', 'map']:
                if key not in metric.keys():
                    raise KeyError('metric must contain "mean" and "map"')

            metric_mean = metric['mean']
            metric_map = metric['map']

        else:
            raise TypeError('metric must be dict or str')

        return metric_mean, metric_map

    def _check_data_dim(self, X):
        """Check data shape and return the size of SPD matrix."""
        shape_X = X.shape
        if len(X.shape) == 2:
            n_channels = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if n_channels != int(n_channels):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(n_channels)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)
        self._check_reference_points(X)

        if self.tsupdate:
            Cr = mean_covariance(X, metric=self.metric_mean)
        else:
            Cr = self.reference_
        return tangent_space(X, Cr, metric=self.metric_map)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)

        self.reference_ = mean_covariance(
            X,
            metric=self.metric_mean,
            sample_weight=sample_weight
        )
        return tangent_space(X, self.reference_, metric=self.metric_map)

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_ts)
            Set of tangent space projections of the matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices corresponding to each of tangent vector.
        """
        self.metric_mean, self.metric_map = self._check_metric(self.metric)
        self._check_reference_points(X)
        return untangent_space(X, self.reference_, metric=self.metric_map)


class FGDA(BaseEstimator, TransformerMixin):

    """Fisher Geodesic Discriminant analysis.

    Project data in Tangent space, apply a FLDA to reduce dimention, and
    project filtered data back in the manifold.
    For a complete description of the algorithm, see [1]_.

    Parameters
    ----------
    metric : string | dict, default='riemann'
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric) and for tangent
        space map (see `tangent_space` for the list of supported metric).
        The metric could be a dict with two keys, `mean` and `map` in
        order to pass different metrics for the reference matrix estimation
        and the tangent space mapping.
    tsupdate : bool, default=False
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]_. This is not compatible with
        online implementation. Performance are better when the number of
        matrices for prediction is higher.

    See Also
    --------
    FgMDM
    TangentSpace

    References
    ----------
    .. [1] `Riemannian geometry applied to BCI classification
        <https://hal.archives-ouvertes.fr/hal-00602700/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. 9th International
        Conference Latent Variable Analysis and Signal Separation
        (LVA/ICA 2010), LNCS vol. 6365, 2010, p. 629-636.
    .. [2] `Classification of covariance matrices using a Riemannian-based
        kernel for BCI applications
        <https://hal.archives-ouvertes.fr/hal-00820475/>`_
        A. Barachant, S. Bonnet, M. Congedo and C. Jutten. Neurocomputing,
        Elsevier, 2013, 112, pp.172-178.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

    def _fit_lda(self, X, y, sample_weight=None):
        """Helper to fit LDA."""
        self.classes_ = np.unique(y)
        self._lda = LDA(n_components=len(self.classes_) - 1,
                        solver='lsqr',
                        shrinkage='auto')

        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self._lda.fit(ts, y)

        W = self._lda.coef_.copy()
        self._W = W.T @ np.linalg.pinv(W @ W.T) @ W
        return ts

    def _retro_project(self, ts):
        """Helper to project back in the manifold."""
        ts = ts @ self._W
        return self._ts.inverse_transform(ts)

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point and the FLDA.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : FGDA instance
            The FGDA instance.
        """
        self._ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Filtering operation.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        covs : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices after filtering.
        """
        ts = self._ts.transform(X)
        return self._retro_project(ts)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        covs : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices after filtering.
        """
        self._ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)


class TangentSpaceSPDxSiegelDisk(BaseEstimator, TransformerMixin):

    """Tangent space project TransformerMixin for SPD x Siegel Disk.

    Tangent space projection map a set of SPD matrices to their
    tangent space according to [1]_ and the Siegel Disk matrix according to [2].

    Tangent space projection is useful to convert SPD matrices in
    Euclidean vectors while conserving the inner structure of the manifold.
    After projection, standard processing and vector-based classification can
    be applied.

    Tangent space projection is a local approximation of the manifold. it takes
    one parameter, the reference point, that is usually estimated using the
    geometric mean of the SPD matrices set you project. If the function
    `fit` is not called, the identity matrix will be used as reference point.
    This can lead to serious degradation of performances.
    The approximation will be bigger if the matrices in the set are scattered
    in the manifold, and lower if they are grouped in a small region of the
    manifold.

    After projection, it is possible to go back in the manifold using the
    inverse transform.

    Parameters
    ----------
    metric : string | dict, default=['riemann', 'siegel']
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric) and for tangent
        space map (see `tangent_space` for the list of supported metric).
        The metric could be a dict with two keys, `mean` and `map` in
        order to pass different metrics for the reference matrix estimation
        and the tangent space mapping.

    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian Geometry
            <https://hal.archives-ouvertes.fr/hal-00681328>`_
            A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
            on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    .. [2] Cabanes, Yann, and Frank Nielsen. "Classification in the Siegel Space for Vectorial
           Autoregressive Data." Geometric Science of Information: 5th International Conference,
           GSI 2021, Paris, France, July 21–23, 2021, Proceedings 5. Springer International Publishing, 2021.
    """

    def __init__(self, metric=['riemann', 'siegel'], alpha=1):
        """Init."""
        self.metric = metric
        self.alpha = alpha

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        self.metric_mean_SPD, self.metric_map_SPD, self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(self.metric)

        references = np.empty((X.shape[1], X.shape[2], X.shape[3]))

        references[0, :, :] = np.real(mean_covariance(
            X[:, 0],
            metric=self.metric_mean_SPD,
            sample_weight=sample_weight
        ))

        for i in np.arange(1, X.shape[1]):
            references[i, :, :] = np.real(mean_covariance_siegel(
                X[:, i],
                metric=self.metric_mean_Siegel,
                sample_weight=sample_weight
            ))

        self.reference_ = np.real(references)
        return self

    def _check_metric(self, metric):

        if isinstance(metric, list):
            if len(metric) != 2:
                raise TypeError('metric must be 2 dim list')

        metric_mean_SPD = metric[0]
        metric_map_SPD = metric[0]
        metric_mean_Siegel = metric[1]
        metric_map_Siegel = metric[1]

        return metric_mean_SPD, metric_map_SPD, metric_mean_Siegel, metric_map_Siegel

    def _check_data_dim(self, X):
        """Check data shape and return the size of SPD matrix."""
        """shape_X = X.shape
        if len(X.shape) == 2:
            n_channels = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if n_channels != int(n_channels):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(n_channels)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")"""

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        """if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')"""

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean_SPD, self.metric_map_SPD, self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(self.metric)
        # self._check_reference_points(X)

        Cr = self.reference_

        tangent = np.empty_like(X)
        tan_ = np.real(tangent_space(X[:, 0, :, :], Cr[0], metric=self.metric_map_SPD))
        tangent[:, 0] = self.alpha * unupper(tan_)

        for i in np.arange(1, X.shape[1]):
            tangent[:, i] = np.real(tangent_space_siegel(
                X[:, i, :, :],
                Cr[i],
                metric=self.metric_map_Siegel))

        return np.real(tangent.reshape(X.shape[0], -1))

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean_SPD, self.metric_map_SPD, self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(
            self.metric)

        references = np.empty((X.shape[1], X.shape[2], X.shape[3]))

        references[0, :, :] = np.real(mean_covariance(
            X[:, 0],
            metric=self.metric_mean_SPD,
            sample_weight=sample_weight
        ))

        for i in np.arange(1, X.shape[1]):
            references[i, :, :] = np.real(mean_covariance_siegel(
                X[:, i],
                metric=self.metric_mean_Siegel,
                sample_weight=sample_weight
            ))

        self.reference_ = references

        tangent = np.empty_like(X)
        tan_ = np.real(tangent_space(X[:, 0, :, :], self.reference_[0], metric=self.metric_map_SPD))
        tangent[:, 0] = self.alpha * unupper(tan_)

        for i in np.arange(1, X.shape[1]):
            tangent[:, i] = np.real(tangent_space_siegel(
                X[:, i, :, :],
                self.reference_[i],
                metric=self.metric_map_Siegel))

        return np.real(tangent.reshape(X.shape[0], -1))

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_ts)
            Set of tangent space projections of the matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices corresponding to each of tangent vector.
        """
        """self.metric_mean_SPD, self.metric_map_SPD, self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(
            self.metric)
        #self._check_reference_points(X)
        # TODO
        return untangent_space(X, self.reference_, metric=self.metric_map)"""


class TangentSpaceSiegelDisk(BaseEstimator, TransformerMixin):

    """Tangent space project TransformerMixin for Siegel Disk.

    Tangent space projection map the Siegel Disk matrix according to [1].

    Tangent space projection is useful to convert SPD matrices in
    Euclidean vectors while conserving the inner structure of the manifold.
    After projection, standard processing and vector-based classification can
    be applied.

    Tangent space projection is a local approximation of the manifold. it takes
    one parameter, the reference point, that is usually estimated using the
    geometric mean of the SPD matrices set you project. If the function
    `fit` is not called, the identity matrix will be used as reference point.
    This can lead to serious degradation of performances.
    The approximation will be bigger if the matrices in the set are scattered
    in the manifold, and lower if they are grouped in a small region of the
    manifold.

    After projection, it is possible to go back in the manifold using the
    inverse transform.

    Parameters
    ----------
    metric : string | dict, default=['riemann', 'siegel']
        The type of metric used for reference matrix estimation (see
        `mean_covariance` for the list of supported metric) and for tangent
        space map (see `tangent_space` for the list of supported metric).
        The metric could be a dict with two keys, `mean` and `map` in
        order to pass different metrics for the reference matrix estimation
        and the tangent space mapping.

    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.

    References
    ----------
    .. [1] Cabanes, Yann, and Frank Nielsen. "Classification in the Siegel Space for Vectorial
           Autoregressive Data." Geometric Science of Information: 5th International Conference,
           GSI 2021, Paris, France, July 21–23, 2021, Proceedings 5. Springer International Publishing, 2021.
    """

    def __init__(self, metric=['siegel'], alpha=1):
        """Init."""
        self.metric = metric
        self.alpha = alpha

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(self.metric)

        references = np.empty((X.shape[1], X.shape[2], X.shape[3]))


        for i in np.arange(0, X.shape[1]):
            references[i, :, :] = np.real(mean_covariance_siegel(
                X[:, i],
                metric=self.metric_mean_Siegel,
                sample_weight=sample_weight
            ))

        self.reference_ = np.real(references)
        return self

    def _check_metric(self, metric):

        metric_mean_Siegel = metric[0]
        metric_map_Siegel = metric[0]

        return metric_mean_Siegel, metric_map_Siegel

    def _check_data_dim(self, X):
        """Check data shape and return the size of SPD matrix."""
        """shape_X = X.shape
        if len(X.shape) == 2:
            n_channels = (np.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if n_channels != int(n_channels):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(n_channels)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")"""

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        """if not hasattr(self, 'reference_'):
            self.reference_ = np.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')"""

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(self.metric)
        # self._check_reference_points(X)

        Cr = self.reference_

        tangent = np.empty_like(X)

        for i in np.arange(0, X.shape[1]):
            tangent[:, i] = np.real(tangent_space_siegel(
                X[:, i, :, :],
                Cr[i],
                metric=self.metric_map_Siegel))

        return np.real(tangent.reshape(X.shape[0], -1))

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        ts : ndarray, shape (n_matrices, n_ts)
            Tangent space projections of SPD matrices.
        """
        self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(self.metric)

        references = np.empty((X.shape[1], X.shape[2], X.shape[3]))

        for i in np.arange(0, X.shape[1]):
            references[i, :, :] = np.real(mean_covariance_siegel(
                X[:, i],
                metric=self.metric_mean_Siegel,
                sample_weight=sample_weight
            ))

        self.reference_ = references

        tangent = np.empty_like(X)

        for i in np.arange(0, X.shape[1]):
            tangent[:, i] = np.real(tangent_space_siegel(
                X[:, i, :, :],
                self.reference_[i],
                metric=self.metric_map_Siegel))

        return np.real(tangent.reshape(X.shape[0], -1))

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_ts)
            Set of tangent space projections of the matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices corresponding to each of tangent vector.
        """
        """self.metric_mean_SPD, self.metric_map_SPD, self.metric_mean_Siegel, self.metric_map_Siegel = self._check_metric(
            self.metric)
        #self._check_reference_points(X)
        # TODO
        return untangent_space(X, self.reference_, metric=self.metric_map)"""