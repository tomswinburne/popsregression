# Backwards compatibility shim. Remove in a future release.
import warnings

warnings.warn(
    "Importing from 'POPSRegression' is deprecated. "
    "Use 'from popsregression import POPSRegression' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from popsregression import POPSRegression  # noqa: F401, E402

__all__ = ["POPSRegression"]
