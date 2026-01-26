# POPSRegression

**[Try the online demo](https://kermodegroup.github.io/demos/apps/regression-demo.html) from Prof. James Kermode (U Warwick) 

Linear regression scheme from the paper 

*Parameter uncertainties for imperfect surrogate models in the low-noise regime*

TD Swinburne and D Perez, [Machine Learning: Science and Technology 2025](http://iopscience.iop.org/article/10.1088/2632-2153/ad9fce)

```bibtex
@article{swinburne2025,
	author={Swinburne, Thomas and Perez, Danny},
	title={Parameter uncertainties for imperfect surrogate models in the low-noise regime},
	journal={Machine Learning: Science and Technology},
	doi={10.1088/2632-2153/ad9fce},
	year={2025}
}
```

## Installation
There will be a PR on `scikit-learn` "soon", but in the meantime
```bash
pip install POPSRegression
```

## What is POPSRegression?

**Bayesian regression for low-noise data (vanishing aleatoric uncertainty).**

Fits the weights of a regression model using BayesianRidge, then estimates weight uncertainties (`sigma_` in `BayesianRidge`) accounting for model misspecification using the POPS (Pointwise Optimal Parameter Sets) algorithm [1]. The `alpha_` attribute which estimates aleatoric uncertainty is not used for predictions as correctly it should be assumed negligable.

Bayesian regression is often used in computational science to fit the weights of a surrogate model which approximates some complex calculation. 
In many important cases the target calculation is near-deterministic, or low-noise, meaning the true data has vanishing aleatoric uncertainty. However, there can be large misspecification uncertainty, i.e. the model weights are instrinsically uncertain as the model is unable to exactly match training data. 

Existing Bayesian regression schemes based on loss minimization can only estimate epistemic and aleatoric uncertainties. In the low-noise limit, weight uncertainties (`sigma_` in `BayesianRidge`) are significantly underestimated as they only account for epistemic uncertainties which decay with increasing data. Predictions then assume any additional error is due to an aleatoric uncertainty (`alpha_` in `BayesianRidge`), which is erroneous in a low-noise setting. This has significant implications on how uncertainty is propagated using weight uncertainties. 

## Example usage
Here, usage follows `sklearn.linear_model`, inheriting `BayesianRidge`

After running `BayesianRidge.fit(..)`, the `alpha_` attribute is not used for predictions.

The `sigma_` matrix still contains epistemic weight uncertainties, whilst `misspecification_sigma_` contains the POPS uncertainties. 

```python

from POPSRegression import POPSRegression

X_train,X_test,y_train,y_test = ...

# Sobol resampling of hypercube with 1.0 samples / training point
model = POPSRegression(resampling_method='sobol',resample_density=1.)

# fit the model, sample POPS hypercube
model.fit(X_train,y_train)

# Return mean and hypercube std
y_pred, y_std = model.predict(X_test,return_std=True)

# can also return max/min 
y_pred, y_std, y_max, y_min = model.predict(X_test,return_std=True,return_bounds=True)

# can also return the epistemic uncertainty seperately
y_pred, y_std, y_max, y_min, y_epistemic_std = model.predict(X_test,return_std=True,return_bounds=True,return_epistemic_std=True)
```

# Toy example
Extreme low-dimensional case, fitting N data points to a quartic polynomial (P=5 parameters) to a complex oscillatory function.
Green: two sigma of `sigma_` weight uncertainty from Bayesian Regression (i.e. without `alpha_` term for aleatoric error)
Orange: two sigma of `sigma_` and `misspecification_sigma_` posterior from POPS Regression
Gray: min-max of posterior from POPS Regression

As can be seen, the final error bars give very good coverage of the test output
<img src="https://github.com/tomswinburne/POPS-Regression/blob/main/example_image.png?raw=true"></img>
