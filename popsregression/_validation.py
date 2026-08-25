"""Constraint-based validation of estimator parameters.

A minimal standalone replacement for ``sklearn.utils._param_validation``,
supporting exactly the constraint kinds used by this package. Estimators
declare a ``_parameter_constraints`` mapping and decorate ``fit`` with
``_fit_context``; the constructor arguments are then checked on every call.

Adapted from scikit-learn (BSD-3-Clause).
"""

# Authors: Thomas D Swinburne <tswin@umich.edu>
#          Danny Perez <danny_perez@lanl.gov>
# SPDX-License-Identifier: BSD-3-Clause

import functools
from abc import ABC, abstractmethod
from numbers import Integral, Real

import numpy as np

__all__ = [
    "Interval",
    "InvalidParameterError",
    "Options",
    "StrOptions",
    "_fit_context",
    "validate_parameter_constraints",
]


class InvalidParameterError(ValueError, TypeError):
    """Raised when an estimator parameter has an invalid type or value.

    Inherits from both ``ValueError`` and ``TypeError`` so that either can be
    caught, matching the behaviour callers expect from scikit-learn.
    """


class _Constraint(ABC):
    """Base class for the constraint objects."""

    @abstractmethod
    def is_satisfied_by(self, val):
        """Whether ``val`` satisfies this constraint."""

    @abstractmethod
    def __str__(self):
        """A human-readable description, used in error messages."""


class _InstancesOf(_Constraint):
    """Constraint satisfied by any instance of ``type``."""

    def __init__(self, type):
        self.type = type

    def is_satisfied_by(self, val):
        # bool is a subclass of int, but a boolean is never a valid number here.
        if self.type in (Real, Integral) and isinstance(val, bool):
            return False
        return isinstance(val, self.type)

    def __str__(self):
        qualname = getattr(self.type, "__qualname__", str(self.type))
        return f"an instance of {qualname!r}"


class _NoneConstraint(_Constraint):
    """Constraint satisfied only by ``None``."""

    def is_satisfied_by(self, val):
        return val is None

    def __str__(self):
        return "None"


class _Booleans(_Constraint):
    """Constraint satisfied by a boolean."""

    def is_satisfied_by(self, val):
        return isinstance(val, (bool, np.bool_))

    def __str__(self):
        return "a boolean"


class _ArrayLikes(_Constraint):
    """Constraint satisfied by anything that converts to a numeric array."""

    def is_satisfied_by(self, val):
        if isinstance(val, (str, bytes, dict)):
            return False
        return hasattr(val, "__len__") or hasattr(val, "__array__")

    def __str__(self):
        return "an array-like"


class _RandomStates(_Constraint):
    """Constraint satisfied by a valid ``random_state`` seed."""

    def is_satisfied_by(self, val):
        if val is None or isinstance(val, (np.random.RandomState, np.random.Generator)):
            return True
        return isinstance(val, Integral) and not isinstance(val, bool)

    def __str__(self):
        return "an int, a RandomState instance or None"


class Interval(_Constraint):
    """Constraint satisfied by a number within an interval.

    Parameters
    ----------
    type : {numbers.Real, numbers.Integral}
        The kind of number the value must be.

    left : float or None
        The lower bound; None for an unbounded interval.

    right : float or None
        The upper bound; None for an unbounded interval.

    closed : {'left', 'right', 'both', 'neither'}
        Which of the bounds are included in the interval.
    """

    def __init__(self, type, left, right, *, closed):
        if closed not in ("left", "right", "both", "neither"):
            raise ValueError(
                "'closed' must be one of 'left', 'right', 'both' or 'neither'. "
                f"Got {closed!r} instead."
            )
        if left is not None and right is not None and left > right:
            raise ValueError(
                f"right can't be less than left. Got left={left} and right={right}"
            )
        self.type = type
        self.left = left
        self.right = right
        self.closed = closed

    def is_satisfied_by(self, val):
        if isinstance(val, bool):
            return False
        if self.type is Integral:
            if not isinstance(val, Integral):
                return False
        elif not isinstance(val, Real):
            return False
        if np.isnan(val):
            return False
        if self.left is not None:
            if self.closed in ("left", "both"):
                if val < self.left:
                    return False
            elif val <= self.left:
                return False
        if self.right is not None:
            if self.closed in ("right", "both"):
                if val > self.right:
                    return False
            elif val >= self.right:
                return False
        return True

    def __str__(self):
        kind = "an int" if self.type is Integral else "a float"
        left_bracket = "[" if self.closed in ("left", "both") else "("
        right_bracket = "]" if self.closed in ("right", "both") else ")"
        left = "-inf" if self.left is None else self.left
        right = "inf" if self.right is None else self.right
        return f"{kind} in the range {left_bracket}{left}, {right}{right_bracket}"


class Options(_Constraint):
    """Constraint satisfied by one of a finite set of values of a given type."""

    def __init__(self, type, options):
        self.type = type
        self.options = set(options)

    def is_satisfied_by(self, val):
        return isinstance(val, self.type) and val in self.options

    def __str__(self):
        opts = ", ".join(repr(o) for o in sorted(self.options, key=repr))
        return f"a {self.type.__name__} among {{{opts}}}"


class StrOptions(_Constraint):
    """Constraint satisfied by one of a finite set of strings."""

    def __init__(self, options):
        self.options = set(options)

    def is_satisfied_by(self, val):
        return isinstance(val, str) and val in self.options

    def __str__(self):
        opts = ", ".join(repr(o) for o in sorted(self.options))
        return f"a str among {{{opts}}}"


def make_constraint(constraint):
    """Convert a constraint declaration into a ``_Constraint`` instance."""
    if isinstance(constraint, _Constraint):
        return constraint
    if constraint is None:
        return _NoneConstraint()
    if isinstance(constraint, str):
        if constraint == "boolean":
            return _Booleans()
        if constraint == "array-like":
            return _ArrayLikes()
        if constraint == "random_state":
            return _RandomStates()
        raise ValueError(f"Unknown constraint alias: {constraint!r}")
    if isinstance(constraint, type):
        return _InstancesOf(constraint)
    raise ValueError(f"Unknown constraint type: {constraint!r}")


def validate_parameter_constraints(parameter_constraints, params, caller_name):
    """Check that each parameter satisfies at least one of its constraints.

    Parameters
    ----------
    parameter_constraints : dict
        Maps a parameter name to its list of constraint declarations.
        A value of ``"no_validation"`` skips the parameter.

    params : dict
        Maps a parameter name to the value to validate.

    caller_name : str
        Name shown in the error message, normally the estimator class name.

    Raises
    ------
    InvalidParameterError
        If a parameter satisfies none of its constraints.
    """
    for param_name, param_val in params.items():
        if param_name not in parameter_constraints:
            continue
        constraints = parameter_constraints[param_name]
        if constraints == "no_validation":
            continue

        constraints = [make_constraint(c) for c in constraints]
        if any(c.is_satisfied_by(param_val) for c in constraints):
            continue

        shown = [str(c) for c in constraints]
        if len(shown) == 1:
            expected = shown[0]
        else:
            expected = f"{', '.join(shown[:-1])} or {shown[-1]}"

        raise InvalidParameterError(
            f"The {param_name!r} parameter of {caller_name} must be"
            f" {expected}. Got {param_val!r} instead."
        )


def _fit_context(*, prefer_skip_nested_validation=True):
    """Decorator validating an estimator's parameters before it fits.

    Parameters
    ----------
    prefer_skip_nested_validation : bool, default=True
        Accepted for signature compatibility with scikit-learn. This package
        has no nested-validation context to manage, so it has no effect.

    Returns
    -------
    decorator : callable
        Decorator to apply to a ``fit`` method.
    """

    def decorator(fit_method):
        @functools.wraps(fit_method)
        def wrapper(estimator, *args, **kwargs):
            estimator._validate_params()
            return fit_method(estimator, *args, **kwargs)

        return wrapper

    return decorator
