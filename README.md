popsregression
=================================================

**Modified fork of [popsregression](https://pops-uq.github.io), implementing PAC-Bayes regularization**

## Installation
**This repo has lo
```bash
# clone repository
cd /this/repository
pip install -e .
```

**Dependencies**: scikit-learn >= 1.6.1, scipy >= 1.6.0, numpy >= 1.20.0

## Quick start

```python
from popsregression import POPSRegression

X_train, X_test, y_train, y_test = ...

# Fit POPSRegression (fit_intercept=False by default)
model = POPSRegression(posterior="ellipsoid", pac_bayes=True)
model.fit(X_train, y_train)

# Prediction with misspecification-aware std and max/min
y_pred, y_std, y_max, y_min = model.predict(
    X_test, return_std=True, return_bounds=True
)
```

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `posterior` | `'hypercube'` | Posterior form: `'hypercube'` (PCA-aligned box), `'ensemble'` (raw corrections) or `'ellipsoid'` (uniform ellipsoid, see below) |
| `pac_bayes` | `False` | Add the closed-form PAC-Bayes layer; requires `posterior='ellipsoid'` |
| `posterior_options` | `None` | Settings for the `'ellipsoid'` posterior, e.g. `{'rank': 16}` |
| `random_state` | `None` | Seed for the posterior resampling; `None` uses the global NumPy state |
| `resampling_method` | `'uniform'` | Sampling method: `'uniform'`, `'sobol'`, `'latin'`, `'halton'` |
| `resample_density` | `1.0` | Number of posterior samples per training point |
| `minimum_relative_error` | `0.01` | Relative residual threshold: only points with \|y - Xw\| >= this times the fit RMSE contribute to the POPS posterior |
| `mode_threshold` | `1e-8` | Eigenvalue threshold for hypercube dimensionality |
| `percentile_clipping` | `0.0` | Percentile to clip from hypercube bounds (0–50) |


## Key attributes (after fitting)

| Attribute | Description |
|---|---|
| `coef_` | Regression coefficients (posterior mean) |
| `sigma_` | Epistemic variance-covariance matrix |
| `misspecification_sigma_` | Misspecification variance-covariance matrix from POPS |
| `posterior_samples_` | Samples from the POPS posterior |
| `ellipsoid_` | The fitted ellipsoid; only with `posterior='ellipsoid'` |
| `bound_`, `kl_`, `empirical_H_`, `gamma_` | PAC-Bayes certificate; only with `pac_bayes=True` |
| `alpha_` | Estimated noise precision (not used for prediction) |

| Parameter | Default | Description |
|---|---|---|
| `rank` | `32` | Rank of the low-rank ellipsoid update `B = B0 + U U^T` |
| `delta` | `1e-3` | Aleatoric width floor added (squared) to predictive widths |
| `baseline` | `'pops'` | Fixed baseline `B0`: `'pops'`, `'ridge'`, or `'zero'` |
| `optimize_center` | `False` | Freeze the mean at the POPS pre-fit; `True` optimizes it jointly |
| `rho_schedule` | `(1e-1, ..., 1e-4)` | Continuation schedule of the log-barrier |

These, and the PAC-Bayes settings (`hyperprior_center`, `hyperprior_scale`,
`update_hyperprior`, `bound_xi`, ...), go through
`POPSRegression(posterior='ellipsoid', posterior_options={...})`. `pac_bayes`,
`fit_intercept` and `random_state` are set on the estimator itself, and sample
weights are passed to `fit`.

See [Ellipsoid posteriors](https://POPS-UQ.github.io/popsregression/ellipse/)
in the documentation, and
[examples/example_polynomial.py](examples/example_polynomial.py), for details.

## Pipeline compatibility

`POPSRegression` is fully compatible with scikit-learn pipelines and
hyperparameter search:

```python
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(
    PolynomialFeatures(degree=4),
    POPSRegression(resampling_method='sobol'),
)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
```

## Documentation

https://POPS-UQ.github.io/popsregression

## Development

The repository is managed with [uv](https://docs.astral.sh/uv/); `uv run`
resolves the pinned environment from `uv.lock` on first use.

```bash
uv run --group test pytest -vsl popsregression        # tests
uv run --group lint ruff check popsregression examples  # linter
uv run --group lint black --check popsregression examples
uv run --group doc mkdocs serve                       # docs at localhost:8000
uv run --extra examples examples/example_polynomial.py  # example figures
```
Without uv, `pip install -e ".[examples]"` and run the tools directly.

## AI Usage
Claude was used to produce full documentation and some test cases. 
All code was reviewed, tested and approved by humans
