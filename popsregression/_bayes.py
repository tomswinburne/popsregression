"""Bayesian ridge regression.

The evidence-maximization linear model that POPS builds on, vendored so this
package depends only on numpy and scipy. The algorithm is scikit-learn's
:class:`sklearn.linear_model.BayesianRidge`, reproduced unchanged so results
match it exactly; only the sparse-input and array-API branches are dropped,
since this package accepts dense numpy input only.

Adapted from scikit-learn (BSD-3-Clause).
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

from math import log
from numbers import Integral, Real

import numpy as np
from scipy import linalg

from ._base import (
    BaseEstimator,
    RegressorMixin,
    _check_sample_weight,
    validate_data,
)
from ._validation import Interval, _fit_context

__all__ = ["BayesianRidge", "LinearModel", "_preprocess_data", "_rescale_data"]


def _preprocess_data(
    X, y, *, fit_intercept, copy=True, copy_y=True, sample_weight=None
):
    """Center ``X`` and ``y`` for a linear model fit.

    When ``fit_intercept`` is True both are centered, optionally weighted by
    ``sample_weight``, and the offsets are returned so the intercept can be
    recovered afterwards. When it is False no centering happens and the
    offsets are zero.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Design matrix.

    y : ndarray of shape (n_samples,) or (n_samples, n_targets)
        Targets.

    fit_intercept : bool
        Whether to center the data.

    copy : bool, default=True
        Whether to copy ``X`` rather than centering it in place.

    copy_y : bool, default=True
        Whether to copy ``y`` rather than centering it in place.

    sample_weight : ndarray of shape (n_samples,), default=None
        Weights used when computing the offsets.

    Returns
    -------
    X_out : ndarray
    y_out : ndarray
    X_offset : ndarray of shape (n_features,)
    y_offset : float or ndarray
    X_scale : ndarray of shape (n_features,)
        Always ones; kept for signature compatibility.
    """
    n_features = X.shape[1]
    if isinstance(sample_weight, Real):
        sample_weight = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)

    X = np.array(X, dtype=X.dtype, order="K", copy=copy)
    y = np.array(y, dtype=X.dtype, copy=copy_y)
    dtype_ = X.dtype

    if fit_intercept:
        X_offset = np.average(X, axis=0, weights=sample_weight).astype(dtype_)
        X -= X_offset
        y_offset = np.average(y, axis=0, weights=sample_weight)
        y -= y_offset
    else:
        X_offset = np.zeros(n_features, dtype=dtype_)
        if y.ndim == 1:
            y_offset = np.asarray(0.0, dtype=dtype_)
        else:
            y_offset = np.zeros(y.shape[1], dtype=dtype_)

    X_scale = np.ones(n_features, dtype=dtype_)
    return X, y, X_offset, y_offset, X_scale


def _rescale_data(X, y, sample_weight):
    """Rescale ``X`` and ``y`` by the square root of ``sample_weight``.

    This turns a weighted least-squares problem into an ordinary one, since
    ``(y - Xw)' S (y - Xw)`` becomes ``||sqrt(S) y - sqrt(S) X w||^2``.

    Returns
    -------
    X_rescaled : ndarray
    y_rescaled : ndarray
    sample_weight_sqrt : ndarray
    """
    sample_weight_sqrt = np.sqrt(sample_weight)
    X = X * sample_weight_sqrt[:, None]
    if y.ndim == 1:
        y = y * sample_weight_sqrt
    else:
        y = y * sample_weight_sqrt[:, None]
    return X, y, sample_weight_sqrt


class LinearModel:
    """Mixin supplying the prediction and intercept handling of a linear model."""

    def _decision_function(self, X):
        """Return ``X @ coef_ + intercept_`` after validating ``X``."""
        from ._base import check_is_fitted

        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], reset=False)
        return X @ self.coef_.T + self.intercept_

    def _set_intercept(self, X_offset, y_offset, X_scale):
        """Recover the intercept from the offsets used to center the data."""
        if self.fit_intercept:
            self.coef_ = np.divide(self.coef_, X_scale)
            self.intercept_ = y_offset - np.dot(X_offset, self.coef_.T)
        else:
            self.intercept_ = np.float64(0.0)


class BayesianRidge(RegressorMixin, LinearModel, BaseEstimator):
    """Bayesian ridge regression.

    Fits a linear model and estimates the noise precision ``alpha_`` and the
    weight precision ``lambda_`` by maximizing the marginal likelihood,
    following MacKay's evidence framework.

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations of the convergence loop.

    tol : float, default=1e-3
        Stop once the coefficient vector has converged to this tolerance.

    alpha_1 : float, default=1e-6
        Shape parameter of the Gamma prior over ``alpha_``.

    alpha_2 : float, default=1e-6
        Inverse scale (rate) parameter of the Gamma prior over ``alpha_``.

    lambda_1 : float, default=1e-6
        Shape parameter of the Gamma prior over ``lambda_``.

    lambda_2 : float, default=1e-6
        Inverse scale (rate) parameter of the Gamma prior over ``lambda_``.

    alpha_init : float, default=None
        Initial value for ``alpha_``. If None, ``1 / Var(y)`` is used.

    lambda_init : float, default=None
        Initial value for ``lambda_``. If None, 1 is used.

    compute_score : bool, default=False
        Whether to record the log marginal likelihood at each iteration.

    fit_intercept : bool, default=True
        Whether to center the data and fit an intercept.

    copy_X : bool, default=True
        If True, ``X`` will be copied; else, it may be overwritten.

    verbose : bool, default=False
        Verbose mode when fitting the model.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,)
        Posterior mean of the weights.

    intercept_ : float
        Independent term of the linear model.

    alpha_ : float
        Estimated precision of the noise.

    lambda_ : float
        Estimated precision of the weights.

    sigma_ : ndarray of shape (n_features, n_features)
        Posterior covariance of the weights.

    scores_ : ndarray
        Log marginal likelihood per iteration; only if ``compute_score=True``.

    n_iter_ : int
        Number of iterations run before convergence.

    n_features_in_ : int
        Number of features seen during fit.
    """

    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="neither")],
        "lambda_init": [None, Interval(Real, 0, None, closed="neither")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
    }

    def __init__(
        self,
        *,
        max_iter=300,
        tol=1.0e-3,
        alpha_1=1.0e-6,
        alpha_2=1.0e-6,
        lambda_1=1.0e-6,
        lambda_2=1.0e-6,
        alpha_init=None,
        lambda_init=None,
        compute_score=False,
        fit_intercept=True,
        copy_X=True,
        verbose=False,
    ):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.alpha_init = alpha_init
        self.lambda_init = lambda_init
        self.compute_score = compute_score
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.verbose = verbose

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X, y = validate_data(
            self,
            X,
            y,
            dtype=[np.float64, np.float32],
            y_numeric=True,
        )
        dtype = X.dtype

        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=dtype)

        X, y, X_offset_, y_offset_, X_scale_ = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        if sample_weight is not None:
            # Sample weight can be implemented via a simple rescaling.
            X, y, _ = _rescale_data(X, y, sample_weight)

        self.X_offset_ = X_offset_
        self.X_scale_ = X_scale_
        n_samples, n_features = X.shape

        # Initialization of the values of the parameters
        eps = np.finfo(np.float64).eps
        # Add `eps` in the denominator to omit division by zero if `np.var(y)`
        # is zero.
        alpha_ = self.alpha_init
        lambda_ = self.lambda_init
        if alpha_ is None:
            alpha_ = 1.0 / (np.var(y) + eps)
        if lambda_ is None:
            lambda_ = 1.0

        # Avoid unintended type promotion to float64 with numpy 2
        alpha_ = np.asarray(alpha_, dtype=dtype)
        lambda_ = np.asarray(lambda_, dtype=dtype)

        verbose = self.verbose
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2

        self.scores_ = list()
        coef_old_ = None

        XT_y = np.dot(X.T, y)
        U, S, Vh = linalg.svd(X, full_matrices=False)
        eigen_vals_ = S**2

        # Convergence loop of the bayesian ridge regression
        for iter_ in range(self.max_iter):
            # update posterior mean coef_ based on alpha_ and lambda_ and
            # compute corresponding rmse
            coef_, rmse_ = self._update_coef_(
                X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
            )
            if self.compute_score:
                # compute the log marginal likelihood
                s = self._log_marginal_likelihood(
                    n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_
                )
                self.scores_.append(s)

            # Update alpha and lambda according to (MacKay, 1992)
            gamma_ = np.sum((alpha_ * eigen_vals_) / (lambda_ + alpha_ * eigen_vals_))
            lambda_ = (gamma_ + 2 * lambda_1) / (np.sum(coef_**2) + 2 * lambda_2)
            alpha_ = (n_samples - gamma_ + 2 * alpha_1) / (rmse_ + 2 * alpha_2)

            # Check for convergence
            if iter_ != 0 and np.sum(np.abs(coef_old_ - coef_)) < self.tol:
                if verbose:
                    print("Convergence after ", str(iter_), " iterations")
                break
            coef_old_ = np.copy(coef_)

        self.n_iter_ = iter_ + 1

        # return regularization parameters and corresponding posterior mean,
        # log marginal likelihood and posterior covariance
        self.alpha_ = alpha_
        self.lambda_ = lambda_
        self.coef_, rmse_ = self._update_coef_(
            X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
        )
        if self.compute_score:
            # compute the log marginal likelihood
            s = self._log_marginal_likelihood(
                n_samples, n_features, eigen_vals_, alpha_, lambda_, coef_, rmse_
            )
            self.scores_.append(s)
            self.scores_ = np.array(self.scores_)

        # posterior covariance is given by 1/alpha_ * scaled_sigma_
        scaled_sigma_ = np.dot(
            Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis]
        )
        self.sigma_ = (1.0 / alpha_) * scaled_sigma_

        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def predict(self, X, return_std=False):
        """Predict using the linear model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        return_std : bool, default=False
            Whether to also return the standard deviation of the predictive
            distribution.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            Mean of the predictive distribution.

        y_std : ndarray of shape (n_samples,)
            Standard deviation of the predictive distribution. Only returned
            if ``return_std=True``.
        """
        y_mean = self._decision_function(X)
        if not return_std:
            return y_mean
        sigmas_squared_data = (np.dot(X, self.sigma_) * X).sum(axis=1)
        y_std = np.sqrt(sigmas_squared_data + (1.0 / self.alpha_))
        return y_mean, y_std

    def _update_coef_(
        self, X, y, n_samples, n_features, XT_y, U, Vh, eigen_vals_, alpha_, lambda_
    ):
        """Update the posterior mean and compute the corresponding rmse.

        The posterior mean is ``coef_ = scaled_sigma_ * X.T * y`` where
        ``scaled_sigma_ = (lambda_/alpha_ * eye(n_features) + X.T @ X)^-1``.
        """
        if n_samples > n_features:
            coef_ = np.linalg.multi_dot(
                [Vh.T, Vh / (eigen_vals_ + lambda_ / alpha_)[:, np.newaxis], XT_y]
            )
        else:
            coef_ = np.linalg.multi_dot(
                [X.T, U / (eigen_vals_ + lambda_ / alpha_)[None, :], U.T, y]
            )

        rmse_ = np.sum((y - np.dot(X, coef_)) ** 2)

        return coef_, rmse_

    def _log_marginal_likelihood(
        self, n_samples, n_features, eigen_vals, alpha_, lambda_, coef, rmse
    ):
        """Return the log marginal likelihood of the current parameters."""
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2

        # compute the log of the determinant of the posterior covariance.
        # posterior covariance is given by
        # sigma = (lambda_ * np.eye(n_features) + alpha_ * np.dot(X.T, X))^-1
        if n_samples > n_features:
            logdet_sigma = -np.sum(np.log(lambda_ + alpha_ * eigen_vals))
        else:
            logdet_sigma = np.full(n_features, lambda_, dtype=np.array(lambda_).dtype)
            logdet_sigma[:n_samples] += alpha_ * eigen_vals
            logdet_sigma = -np.sum(np.log(logdet_sigma))

        score = lambda_1 * log(lambda_) - lambda_2 * lambda_
        score += alpha_1 * log(alpha_) - alpha_2 * alpha_
        score += 0.5 * (
            n_features * log(lambda_)
            + n_samples * log(alpha_)
            - alpha_ * rmse
            - lambda_ * np.sum(coef**2)
            + logdet_sigma
            - n_samples * log(2 * np.pi)
        )

        return score
