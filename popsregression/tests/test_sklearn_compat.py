"""Compatibility with scikit-learn.

This package does not depend on scikit-learn, but its estimators follow the
scikit-learn estimator protocol so that they drop into pipelines and
hyper-parameter search for users who do have it installed. Everything that
needs scikit-learn to verify that promise lives here, behind an
``importorskip``, so the rest of the suite runs without it.
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from numpy.testing import assert_allclose

from popsregression import POPSRegression
from popsregression._bayes import BayesianRidge
from popsregression._ellipse import _EllipsoidPosterior

pytest.importorskip("sklearn")

from sklearn.base import clone  # noqa: E402
from sklearn.linear_model import BayesianRidge as SKBayesianRidge  # noqa: E402
from sklearn.model_selection import GridSearchCV  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402


def _make_low_noise_data(n_samples=50, n_features=5, noise=0.01, seed=42):
    """Low-noise polynomial regression data."""
    rng = np.random.RandomState(seed)
    x = np.sort(rng.uniform(-1, 1, n_samples)) * 10
    f = lambda x: (x**3 + 0.01 * x**4) * 0.1 + np.sin(x) * x * 10.0
    X = np.vander(x, n_features, increasing=True)
    y = f(x) + noise * rng.randn(n_samples)
    return X, y


# --- The vendored BayesianRidge reproduces scikit-learn's exactly ---


@pytest.mark.parametrize("shape", [(60, 5), (5, 60), (40, 40)])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("fit_intercept", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_vendored_bayesian_ridge_matches_sklearn(shape, dtype, fit_intercept, weighted):
    """The vendored solver must agree with scikit-learn's to machine precision.

    Both branches of ``_update_coef_`` are covered by the two rectangular
    shapes, and the square case exercises the boundary between them.
    """
    n_samples, n_features = shape
    rng = np.random.RandomState(0)
    X = rng.randn(n_samples, n_features).astype(dtype)
    y = (X @ rng.randn(n_features) + 0.01 * rng.randn(n_samples)).astype(dtype)
    sample_weight = np.linspace(0.5, 2.0, n_samples) if weighted else None

    kwargs = dict(fit_intercept=fit_intercept, compute_score=True)
    expected = SKBayesianRidge(**kwargs).fit(X, y, sample_weight=sample_weight)
    got = BayesianRidge(**kwargs).fit(X, y, sample_weight=sample_weight)

    for attr in ("coef_", "intercept_", "sigma_", "alpha_", "lambda_", "scores_"):
        assert_allclose(
            np.asarray(getattr(got, attr), dtype=float),
            np.asarray(getattr(expected, attr), dtype=float),
            rtol=1e-11,
            atol=1e-13,
            err_msg=f"{attr} differs from scikit-learn",
        )
    assert got.n_iter_ == expected.n_iter_

    assert_allclose(got.predict(X), expected.predict(X), rtol=1e-11, atol=1e-13)


# --- Estimator-protocol compliance ---

# `check_sample_weight_equivalence_on_dense_data` asserts that fitting with
# integer sample weights matches fitting on the correspondingly repeated rows.
# POPS estimates its misspecification posterior from per-training-point
# residuals, so duplicating a row genuinely changes the posterior. The check
# does not apply, and it failed before this package stopped using
# scikit-learn's base classes too.
_EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": (
        "POPS weights the misspecification posterior per training point, so"
        " sample weights are not equivalent to repeating rows."
    )
}


@parametrize_with_checks(
    [POPSRegression()],
    expected_failed_checks=lambda estimator: _EXPECTED_FAILED_CHECKS,
)
def test_pops_regression_sklearn_compatible(estimator, check):
    check(estimator)


@parametrize_with_checks(
    [_EllipsoidPosterior()],
    expected_failed_checks=lambda estimator: _EXPECTED_FAILED_CHECKS,
)
def test_ellipsoid_posterior_sklearn_compatible(estimator, check):
    check(estimator)


# --- clone / get_params / set_params ---


def test_clone_pops_regression():
    model = POPSRegression(
        posterior="ensemble",
        resample_density=2.0,
        minimum_relative_error=0.05,
    )
    assert clone(model).get_params() == model.get_params()


def test_clone_ellipsoid_posterior():
    model = _EllipsoidPosterior(rank=4, baseline="ridge", pac_bayes=True)
    assert clone(model).get_params() == model.get_params()


def test_pac_bayes_is_a_tunable_parameter():
    """pac_bayes is an ordinary parameter: cloned, settable, introspectable."""
    model = POPSRegression(posterior="ellipsoid", pac_bayes=True, random_state=0)
    assert model.get_params()["pac_bayes"] is True
    assert clone(model).get_params() == model.get_params()
    model.set_params(pac_bayes=False)
    assert model.pac_bayes is False


# --- Pipelines and model selection ---


def test_pipeline_polynomial_features():
    x = np.linspace(-2, 2, 40)
    y = np.sin(2 * x) * x
    pipe = make_pipeline(
        PolynomialFeatures(degree=3),
        _EllipsoidPosterior(random_state=0),
    )
    pipe.fit(x.reshape(-1, 1), y)
    assert pipe.predict(x.reshape(-1, 1)).shape == (40,)


def test_pipeline_with_pops_regression():
    """The pipeline snippet documented in the README."""
    x = np.linspace(-2, 2, 60)
    y = np.sin(2 * x) * x
    pipe = make_pipeline(
        PolynomialFeatures(degree=4),
        POPSRegression(resampling_method="sobol", random_state=0),
    )
    pipe.fit(x.reshape(-1, 1), y)
    assert pipe.predict(x.reshape(-1, 1)).shape == (60,)


def test_grid_search():
    """The GridSearchCV snippet documented in docs/usage.md."""
    x = np.linspace(-2, 2, 60)
    y = np.sin(2 * x) * x
    pipe = make_pipeline(
        PolynomialFeatures(degree=4),
        POPSRegression(random_state=0),
    )
    search = GridSearchCV(
        pipe,
        {
            "polynomialfeatures__degree": [2, 3],
            "popsregression__posterior": ["hypercube", "ensemble"],
        },
        cv=3,
    )
    search.fit(x.reshape(-1, 1), y)
    assert search.best_params_["popsregression__posterior"] in {
        "hypercube",
        "ensemble",
    }


# --- POPS against the plain Bayesian baseline ---


def test_pops_wider_uncertainty_than_bayesian_ridge():
    """POPS combined uncertainty should be wider than BayesianRidge
    epistemic-only uncertainty for misspecified low-noise data."""
    X, y = _make_low_noise_data(n_samples=30, n_features=5, noise=0.001)

    pops = POPSRegression().fit(X, y)
    br = SKBayesianRidge(fit_intercept=False).fit(X, y)

    _, pops_std = pops.predict(X, return_std=True)
    br_epistemic_std = np.sqrt(np.sum(np.dot(X, br.sigma_) * X, axis=1))

    assert np.mean(pops_std) > np.mean(br_epistemic_std)
