# Request for inclusion: popsregression (POPS Regression)

## Summary

**popsregression** provides `POPSRegression`, a scikit-learn compatible Bayesian regression estimator for low-noise data that accounts for model misspecification uncertainty using the POPS (Pointwise Optimal Parameter Sets) algorithm.

**Repository**: https://github.com/tomswinburne/POPS-Regression

## What does it do?

Standard Bayesian regression (e.g. `BayesianRidge`) estimates epistemic and aleatoric uncertainties. In the low-noise limit common in computational science — where surrogate models are fit to near-deterministic simulation data — weight uncertainties are significantly underestimated because they only capture epistemic uncertainty, which decays with increasing data. The remaining error is incorrectly attributed to aleatoric noise.

`POPSRegression` corrects this by estimating **misspecification uncertainty** from pointwise optimal parameter sets. It extends `BayesianRidge` with a POPS posterior estimation step that constructs a distribution of plausible weight perturbations, yielding wider and more honest uncertainty estimates that properly cover the true function.

This is relevant to any application where a linear-in-parameters model is fit to low-noise data and the model class cannot perfectly represent the target (e.g. polynomial surrogate models for atomistic simulations, engineering surrogates, reduced-order models).

## Checklist

- [x] **scikit-learn API compliance**: passes all `parametrize_with_checks` tests (84 sklearn common checks)
- [x] **Unit tests**: 149 tests total including doctests, all passing
- [x] **CI/CD**: GitHub Actions for testing (Linux, macOS, Windows), linting, and documentation deployment
- [x] **Documentation**: Sphinx docs with API reference, user guide, and Sphinx-Gallery example
- [x] **Published reference**: Swinburne, T.D. and Perez, D. (2025). "Parameter uncertainties for imperfect surrogate models in the low-noise regime." *Machine Learning: Science and Technology*, 6, 015008. [doi:10.1088/2632-2153/ad9fce](https://doi.org/10.1088/2632-2153/ad9fce)
- [x] **License**: BSD-3-Clause
- [x] **PyPI**: `pip install popsregression`
- [x] **Example**: gallery example comparing POPS vs BayesianRidge uncertainty across training set sizes
- [x] **Pipeline compatible**: works with `sklearn.pipeline.Pipeline` and hyperparameter search

## API overview

```python
from popsregression import POPSRegression

model = POPSRegression(
    posterior='hypercube',           # or 'ensemble'
    resampling_method='sobol',       # or 'uniform', 'latin', 'halton'
    resample_density=1.0,
    leverage_percentile=50.0,
)
model.fit(X_train, y_train)

# Predict with combined (misspecification + epistemic) uncertainty
y_pred, y_std = model.predict(X_test, return_std=True)

# Also return min/max bounds and epistemic-only uncertainty
y_pred, y_std, y_max, y_min, y_ep_std = model.predict(
    X_test, return_std=True, return_bounds=True, return_epistemic_std=True
)
```

## Authors

- Thomas D Swinburne (tswin@umich.edu)
- Danny Perez (danny_perez@lanl.gov)
