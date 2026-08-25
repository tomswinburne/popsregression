"""Estimator base classes and input validation.

A minimal standalone replacement for the parts of ``sklearn.base`` and
``sklearn.utils.validation`` that this package needs. The estimator protocol
(``get_params``/``set_params``/``score``/``__sklearn_tags__``) is reproduced
faithfully, so estimators built on these classes still work with
``sklearn.base.clone``, :class:`~sklearn.pipeline.Pipeline` and
:class:`~sklearn.model_selection.GridSearchCV` for users who have
scikit-learn installed -- without this package importing it.

Only dense array input is supported; this package never accepts sparse
matrices.

Adapted from scikit-learn (BSD-3-Clause).
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import inspect
import sys
import warnings
from collections import defaultdict

import numpy as np

from ._validation import validate_parameter_constraints

__all__ = [
    "BaseEstimator",
    "NotFittedError",
    "RegressorMixin",
    "_check_sample_weight",
    "check_array",
    "check_is_fitted",
    "check_random_state",
    "check_X_y",
    "validate_data",
]


class NotFittedError(ValueError, AttributeError):
    """Raised when an estimator is used before being fitted."""


class DataConversionWarning(UserWarning):
    """Warned when the input data is silently converted during validation."""


_BRIDGED = {}


def _bridged(name, own):
    """Return a class that is both ``own`` and scikit-learn's version of it.

    scikit-learn's own exception and warning hierarchy is what code written
    against scikit-learn catches, and what its estimator checks assert on. So
    that ``except sklearn.exceptions.NotFittedError`` works on this package's
    errors, the raised class is widened to inherit from scikit-learn's too --
    but only when scikit-learn is *already imported*, which is looked up in
    ``sys.modules`` rather than imported. A scikit-learn-free installation
    always gets ``own`` unchanged, and importing this package never pulls
    scikit-learn in.
    """
    module = sys.modules.get("sklearn.exceptions")
    other = getattr(module, name, None) if module is not None else None
    if other is None or issubclass(own, other):
        return own
    key = (name, id(other))
    if key not in _BRIDGED:
        _BRIDGED[key] = type(name, (own, other), {})
    return _BRIDGED[key]


def check_random_state(seed):
    """Turn ``seed`` into a :class:`numpy.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or RandomState
        If None, return the global ``numpy.random`` singleton. If an int,
        return a new RandomState seeded with it. If already a RandomState or
        Generator, return it unchanged.

    Returns
    -------
    rng : RandomState or Generator
        The random number generator to use.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, (np.random.RandomState, np.random.Generator)):
        return seed
    raise ValueError(f"{seed!r} cannot be used to seed a numpy.random.RandomState")


def _check_feature_names_in(estimator, X, *, reset):
    """Record or verify the feature names carried by ``X``."""
    names = None
    if hasattr(X, "columns"):  # a pandas DataFrame, without importing pandas
        columns = list(X.columns)
        if all(isinstance(c, str) for c in columns):
            names = np.asarray(columns, dtype=object)

    if reset:
        if names is not None:
            estimator.feature_names_in_ = names
        elif hasattr(estimator, "feature_names_in_"):
            del estimator.feature_names_in_
        return

    fitted = getattr(estimator, "feature_names_in_", None)
    if fitted is None and names is None:
        return
    if fitted is None and names is not None:
        raise ValueError(
            "X has feature names, but "
            f"{estimator.__class__.__name__} was fitted without feature names"
        )
    if fitted is not None and names is None:
        raise ValueError(
            f"X does not have valid feature names, but {estimator.__class__.__name__}"
            " was fitted with feature names"
        )
    if not np.array_equal(fitted, names):
        raise ValueError(
            "The feature names should match those that were passed during fit.\n"
            "Feature names seen at fit time, yet now missing:\n"
            f"- {sorted(set(fitted) - set(names))}\n"
        )


def check_array(
    X,
    *,
    dtype="numeric",
    copy=False,
    ensure_2d=True,
    ensure_min_samples=1,
    ensure_min_features=1,
    ensure_all_finite=True,
    input_name="X",
):
    """Validate an array and convert it to a numeric ndarray.

    Parameters
    ----------
    X : array-like
        Input to check and convert.

    dtype : 'numeric', type, list of type or None, default='numeric'
        Target dtype. A list gives the acceptable dtypes in preference order;
        if the input matches none of them it is cast to the first. None
        preserves the input dtype.

    copy : bool, default=False
        Whether to force a copy.

    ensure_2d : bool, default=True
        Whether to raise if the input is not two-dimensional.

    ensure_min_samples : int, default=1
        Minimum number of rows required.

    ensure_min_features : int, default=1
        Minimum number of columns required, checked only when ``ensure_2d``.

    ensure_all_finite : bool, default=True
        Whether to raise if the input contains NaN or infinity.

    input_name : str, default='X'
        Name used in error messages.

    Returns
    -------
    X_converted : ndarray
        The validated array.
    """
    if hasattr(X, "sparse") or (hasattr(X, "format") and hasattr(X, "toarray")):
        raise TypeError(
            f"A sparse matrix was passed as {input_name}, but dense data is required."
        )

    # Reject complex input before any cast, which would silently drop the
    # imaginary part instead of failing.
    raw = np.asarray(X) if not hasattr(X, "dtype") else X
    if np.iscomplexobj(raw):
        raise ValueError(f"Complex data not supported\n{raw}\n")

    if dtype == "numeric":
        target_dtype = np.float64
    elif isinstance(dtype, (list, tuple)):
        source = getattr(X, "dtype", None)
        target_dtype = None if source in dtype else dtype[0]
    else:
        target_dtype = dtype

    if target_dtype is None and getattr(raw, "dtype", None) == object:
        # An object array was not in the accepted dtype list; fall back to
        # float, matching what a numeric estimator can actually consume.
        target_dtype = np.float64

    # A TypeError from numpy names the offending element type, which is more
    # informative than anything added here, so it propagates unchanged.
    try:
        array = np.asarray(X, dtype=target_dtype)
    except ValueError as exc:
        raise ValueError(
            f"Unable to convert {input_name} to a numeric array. {exc}"
        ) from exc

    if array.dtype == object or array.dtype.kind in "USV":
        raise ValueError(
            f"{input_name} must be numeric; got dtype {array.dtype!r} instead."
        )
    if array.dtype.kind in "biu":
        array = array.astype(np.float64)

    if ensure_2d:
        if array.ndim == 0:
            raise ValueError(
                f"Expected 2D array, got scalar array instead:\narray={array}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your data has"
                " a single feature or array.reshape(1, -1) if it contains a single"
                " sample."
            )
        if array.ndim == 1:
            raise ValueError(
                f"Expected 2D array, got 1D array instead:\narray={array}.\n"
                "Reshape your data either using array.reshape(-1, 1) if your data has"
                " a single feature or array.reshape(1, -1) if it contains a single"
                " sample."
            )
        if array.ndim > 2:
            raise ValueError(
                f"Found array with dim {array.ndim}, while dim <= 2 is required"
                f" by {input_name}."
            )

    if ensure_all_finite and not np.isfinite(array).all():
        kind = "infinity" if np.isinf(array).any() else "NaN"
        raise ValueError(
            f"Input {input_name} contains {kind}."
            " Estimators do not accept missing values encoded as NaN natively."
        )

    if ensure_min_samples > 0 and array.shape[0] < ensure_min_samples:
        raise ValueError(
            f"Found array with {array.shape[0]} sample(s) (shape={array.shape}) while"
            f" a minimum of {ensure_min_samples} is required by {input_name}."
        )
    if ensure_2d and ensure_min_features > 0 and array.ndim == 2:
        if array.shape[1] < ensure_min_features:
            raise ValueError(
                f"Found array with {array.shape[1]} feature(s) (shape={array.shape})"
                f" while a minimum of {ensure_min_features} is required"
                f" by {input_name}."
            )

    if copy:
        array = np.array(array, dtype=array.dtype, order="K")
    return array


def _check_y(y, *, dtype=None, y_numeric=False, estimator_name=None):
    """Validate a target vector and return it as a 1d numeric array."""
    y = np.asarray(y)

    # A column vector is accepted but ravelled, with the same warning
    # scikit-learn raises, since silently changing the shape is surprising.
    if y.ndim == 2 and y.shape[1] == 1:
        warnings.warn(
            (
                "A column-vector y was passed when a 1d array was expected. Please"
                " change the shape of y to (n_samples,), for example using ravel()."
            ),
            _bridged("DataConversionWarning", DataConversionWarning),
            stacklevel=2,
        )
        y = y.ravel()
    if y.ndim > 1:
        raise ValueError(
            f"y should be a 1d array, got an array of shape {y.shape} instead."
        )
    if y.dtype == object or y.dtype.kind in "USV":
        # Mirror the wording scikit-learn's estimator checks look for.
        try:
            y = y.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unknown label type: {y.dtype!r}. {exc}") from exc

    y = check_array(
        y,
        dtype=np.float64 if y_numeric else dtype,
        ensure_2d=False,
        ensure_all_finite=True,
        input_name="y",
    )
    if y.size == 0:
        name = estimator_name or "this estimator"
        raise ValueError(
            f"0 sample(s) (shape={y.shape}) while a minimum of 1 is required by {name}."
        )
    return y


def check_X_y(X, y, *, dtype="numeric", y_numeric=False, estimator=None, **kwargs):
    """Validate ``X`` and ``y`` jointly and check that their lengths match.

    Returns
    -------
    X_converted : ndarray
    y_converted : ndarray
    """
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    name = None if estimator is None else estimator.__class__.__name__
    X = check_array(X, dtype=dtype, **kwargs)
    y = _check_y(y, y_numeric=y_numeric, estimator_name=name)
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "Found input variables with inconsistent numbers of samples:"
            f" [{X.shape[0]}, {y.shape[0]}]"
        )
    return X, y


def validate_data(
    estimator,
    X="no_validation",
    y="no_validation",
    *,
    reset=True,
    dtype="numeric",
    y_numeric=False,
    **kwargs,
):
    """Validate input and record or check ``n_features_in_``.

    Parameters
    ----------
    estimator : object
        The estimator whose ``n_features_in_`` and ``feature_names_in_``
        attributes are set (``reset=True``) or verified (``reset=False``).

    X : array-like or 'no_validation'
        Input to validate.

    y : array-like or 'no_validation'
        Target to validate. If left at its default only ``X`` is checked.

    reset : bool, default=True
        True when called from ``fit``; False when called from ``predict`` or a
        second pass within ``fit``, in which case the recorded feature count
        is checked instead of overwritten.

    dtype : 'numeric', type, list of type or None, default='numeric'
        Passed through to :func:`check_array`.

    y_numeric : bool, default=False
        Whether to coerce ``y`` to a floating dtype.

    **kwargs : dict
        Extra keyword arguments for :func:`check_array`.

    Returns
    -------
    out : ndarray or tuple of ndarray
        The validated ``X``, or ``(X, y)`` when ``y`` was given.
    """
    no_X = isinstance(X, str) and X == "no_validation"
    no_y = isinstance(y, str) and y == "no_validation"

    if no_X and no_y:
        raise ValueError("Validation should be done on X, y or both.")

    if not no_X:
        _check_feature_names_in(estimator, X, reset=reset)

    if no_y:
        out = check_array(X, dtype=dtype, **kwargs)
    elif no_X:
        out = _check_y(
            y, y_numeric=y_numeric, estimator_name=estimator.__class__.__name__
        )
    else:
        out = check_X_y(
            X, y, dtype=dtype, y_numeric=y_numeric, estimator=estimator, **kwargs
        )

    if not no_X:
        X_checked = out[0] if isinstance(out, tuple) else out
        n_features = X_checked.shape[1] if X_checked.ndim == 2 else 1
        if reset:
            estimator.n_features_in_ = n_features
        else:
            fitted = getattr(estimator, "n_features_in_", None)
            if fitted is not None and n_features != fitted:
                raise ValueError(
                    f"X has {n_features} features, but"
                    f" {estimator.__class__.__name__} is expecting {fitted} features"
                    " as input."
                )
    return out


def check_is_fitted(estimator, attributes=None):
    """Raise :class:`NotFittedError` if the estimator has not been fitted.

    An estimator counts as fitted once it carries at least one attribute
    ending in a single trailing underscore.

    Parameters
    ----------
    estimator : object
        The estimator to check.

    attributes : str, list of str or None, default=None
        Specific attribute name(s) to require. If None, the trailing
        underscore convention is used.
    """
    if isinstance(estimator, type):
        raise TypeError(f"{estimator} is a class, not an instance.")
    if not hasattr(estimator, "fit"):
        raise TypeError(f"{estimator!r} is not an estimator instance.")

    if attributes is not None:
        if isinstance(attributes, str):
            attributes = [attributes]
        fitted = all(hasattr(estimator, attr) for attr in attributes)
    else:
        fitted = [
            v
            for v in vars(estimator)
            if v.endswith("_") and not v.startswith("__") and not v.endswith("__")
        ]

    if not fitted:
        raise _bridged("NotFittedError", NotFittedError)(
            f"This {estimator.__class__.__name__} instance is not fitted yet. Call"
            " 'fit' with appropriate arguments before using this estimator."
        )


def _check_sample_weight(sample_weight, X, *, dtype=None, ensure_non_negative=False):
    """Validate sample weights and return them as a 1d float array.

    Parameters
    ----------
    sample_weight : array-like, scalar or None
        The weights to validate. None yields an array of ones, and a scalar
        yields an array filled with that value.

    X : ndarray
        The design matrix, used for the expected length.

    dtype : dtype or None, default=None
        Output dtype; float64 unless the input is already float32.

    ensure_non_negative : bool, default=False
        Whether to raise if any weight is negative.

    Returns
    -------
    sample_weight : ndarray of shape (n_samples,)
        The validated weights.
    """
    n_samples = X.shape[0]
    if dtype is not None and dtype not in (np.float32, np.float64):
        dtype = np.float64

    if sample_weight is None:
        sample_weight = np.ones(n_samples, dtype=dtype or np.float64)
    elif isinstance(sample_weight, (int, float, np.integer, np.floating)):
        sample_weight = np.full(n_samples, sample_weight, dtype=dtype or np.float64)
    else:
        sample_weight = check_array(
            sample_weight,
            dtype=dtype or np.float64,
            ensure_2d=False,
            input_name="sample_weight",
        )
        if sample_weight.ndim != 1:
            raise ValueError("Sample weights must be 1D array or scalar")
        if sample_weight.shape != (n_samples,):
            raise ValueError(
                f"sample_weight.shape == {sample_weight.shape},"
                f" expected {(n_samples,)}!"
            )

    if ensure_non_negative and np.any(sample_weight < 0):
        raise ValueError("Negative values in data passed to `sample_weight`")
    return sample_weight


class BaseEstimator:
    """Base class providing the estimator parameter protocol.

    Subclasses must take all of their parameters as explicit keyword
    arguments of ``__init__`` and store each one unmodified on ``self`` under
    the same name; ``get_params`` recovers them by introspecting the
    signature. This is the contract that makes ``sklearn.base.clone`` and the
    scikit-learn meta-estimators work on these classes.
    """

    @classmethod
    def _get_param_names(cls):
        """Return the sorted names of this estimator's parameters."""
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            return []
        signature = inspect.signature(init)
        parameters = [
            p
            for p in signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        for p in parameters:
            if p.kind == p.VAR_POSITIONAL:
                raise RuntimeError(
                    "estimators should always specify their parameters in the"
                    " signature of their __init__ (no varargs)."
                    f" {cls} does not follow this convention."
                )
        return sorted(p.name for p in parameters)

    def get_params(self, deep=True):
        """Get the parameters of this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, also return the parameters of any nested estimator held
            as a parameter value.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        out = {}
        for key in self._get_param_names():
            value = getattr(self, key)
            if deep and hasattr(value, "get_params") and not isinstance(value, type):
                for sub_key, sub_value in value.get_params().items():
                    out[f"{key}__{sub_key}"] = sub_value
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        Supports nested parameters with the ``<component>__<parameter>``
        syntax.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            The estimator instance.
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = defaultdict(dict)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if key not in valid_params:
                local = self._get_param_names()
                raise ValueError(
                    f"Invalid parameter {key!r} for estimator {self}. Valid"
                    f" parameters are: {local!r}."
                )
            if delim:
                nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        for key, sub_params in nested_params.items():
            valid_params[key].set_params(**sub_params)
        return self

    def _validate_params(self):
        """Validate the constructor parameters against the constraints.

        Uses this class's ``_parameter_constraints`` mapping. Called by the
        ``_fit_context`` decorator at the start of every ``fit``.
        """
        validate_parameter_constraints(
            self._parameter_constraints,
            self.get_params(deep=False),
            caller_name=self.__class__.__name__,
        )

    def __repr__(self, N_CHAR_MAX=700):
        """Return a repr showing only the parameters left at a non-default value."""
        cls = self.__class__
        try:
            init_params = inspect.signature(cls.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return f"{cls.__name__}()"

        parts = []
        for name in self._get_param_names():
            if name not in init_params:
                continue
            default = init_params[name].default
            try:
                value = getattr(self, name)
            except AttributeError:  # pragma: no cover - defensive
                continue
            if default is not inspect.Parameter.empty and _is_default(value, default):
                continue
            parts.append(f"{name}={value!r}")

        repr_ = f"{cls.__name__}({', '.join(parts)})"
        if len(repr_) > N_CHAR_MAX:
            keep = (N_CHAR_MAX - 5) // 2
            repr_ = f"{repr_[:keep]} ... {repr_[-keep:]}"
        return repr_

    def __sklearn_tags__(self):
        """Return scikit-learn estimator tags.

        Built lazily from scikit-learn's own tag dataclasses so that the tag
        objects are exactly the ones scikit-learn expects. scikit-learn is
        imported only inside this method, which scikit-learn itself is the
        only caller of -- so a scikit-learn-free installation never reaches
        the import.
        """
        from sklearn.utils import InputTags, Tags, TargetTags

        return Tags(
            estimator_type=None,
            target_tags=TargetTags(required=False),
            transformer_tags=None,
            regressor_tags=None,
            classifier_tags=None,
            input_tags=InputTags(sparse=False),
        )


def _is_default(value, default):
    """Whether ``value`` is unchanged from the parameter's default."""
    if value is default:
        return True
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return False
    try:
        return bool(type(value) is type(default) and value == default)
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return False


class RegressorMixin:
    """Mixin adding the regressor protocol to an estimator."""

    _estimator_type = "regressor"

    def score(self, X, y, sample_weight=None):
        """Return the coefficient of determination R² of the prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for ``X``.

        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        score : float
            R² of ``self.predict(X)`` against ``y``. The best possible score
            is 1.0 and it can be arbitrarily negative.
        """
        y_pred = self.predict(X)
        if isinstance(y_pred, tuple):  # pragma: no cover - defensive
            y_pred = y_pred[0]
        y_true = np.asarray(y, dtype=np.float64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()

        if sample_weight is not None:
            weight = np.asarray(sample_weight, dtype=np.float64).ravel()[:, np.newaxis]
        else:
            weight = 1.0

        numerator = (weight * (y_true - y_pred) ** 2).sum()
        denominator = (
            weight * (y_true - np.average(y_true, weights=sample_weight)) ** 2
        ).sum()
        if denominator == 0.0:
            return 0.0 if numerator else 1.0
        return float(1 - numerator / denominator)

    def __sklearn_tags__(self):
        """Return scikit-learn estimator tags marking this as a regressor."""
        from sklearn.utils import RegressorTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()
        tags.target_tags.required = True
        return tags
