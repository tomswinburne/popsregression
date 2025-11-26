import numpy as np
from sklearn.linear_model import BayesianRidge
from scipy import linalg
from scipy.linalg import pinvh, eigh
from sklearn.base import BaseEstimator, RegressorMixin, _fit_context
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.utils.validation import _check_sample_weight, validate_data
from numbers import Real, Integral
from scipy.stats import qmc 


###############################################################################
# POPS (Pointwise Optimal Parameter Sets) regression


class POPSRegression(BayesianRidge):
    """
    Bayesian regression for low-noise data (vanishing aleatoric uncertainty). 

    Fits the weights of a regression model using BayesianRidge, then estimates weight uncertainties (`sigma_` in `BayesianRidge`) accounting for model misspecification using the POPS (Pointwise Optimal Parameter Sets) algorithm [1]. The `alpha_` attribute which estimates aleatoric uncertainty is not used for predictions as correctly it should be assumed negligable.

    Bayesian regression is often used in computational science to fit the weights of a surrogate model which approximates some complex calcualtion. 
    In many important cases the target calcualtion is near-deterministic, or low-noise, meaning the true data has vanishing aleatoric uncertainty. However, there can be large misspecification uncertainty, i.e. the model weights are instrinsically uncertain as the model is unable to exactly match training data. 

    Existing Bayesian regression schemes based on loss minimization can only estimate epistemic and aleatoric uncertainties. In the low-noise limit, 
    weight uncertainties (`sigma_` in `BayesianRidge`) are significantly underestimated as they only account for epistemic uncertainties which decay with increasing data. Predictions then assume any additional error is due to an aleatoric uncertainty (`alpha_` in `BayesianRidge`), which is erroneous in a low-noise setting. This has significant implications on how uncertainty is propagated using weight uncertainties. 

    Parameters
    ----------
    max_iter : int, default=300
        Maximum number of iterations.
    tol : float, default=1e-3
        Stop the algorithm if w has converged.
    alpha_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter.
    alpha_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter.
    lambda_1 : float, default=1e-6
        Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter.
    lambda_2 : float, default=1e-6
        Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter.
    alpha_init : float, default=None
        Initial value for alpha (precision of the noise).
    lambda_init : float, default=None
        Initial value for lambda (precision of the weights).
    compute_score : bool, default=False
        If True, compute the objective function at each step of the model.
    fit_intercept : bool, default=False
        Whether to calculate the intercept for this model.
    copy_X : bool, default=True
        If True, X will be copied; else, it may be overwritten.
    verbose : bool, default=False
        Verbose mode when fitting the model.
    mode_threshold : float, default=1e-3
        Threshold for determining the mode of the posterior distribution.
    resample_density : float, default=1.0
        Density of resampling for the POPS algorithm- number of hypercube per training point. Default is the greater of 0.5 or 100/n_samples.
    resampling_method : str, default='uniform'
        Method of resampling for the POPS algorithm. 
        must be one of 'sobol', 'latin', 'halton', 'grid', or 'uniform'.
    percentile_clipping : float, default=0.0
        Percentile to clip from each end of the distribution when determining the hypercube bounds, i.e. spans [x,100-x]. Value must be between 0 and 50, but in practice should be between 0.0 and 0.5 for robust bounds (i.e. 0% and 0.5%)
    leverage_percentile : float, default=50.0
        To accelerate hypercube fitting, only consider points in leverage range [`leverage_percentile`,100.0]. Default is 50%.
    posterior : str, default='hypercube'
        Form of POPS parameter posterior. 
        must be one of 'hypercube', 'ensemble' or 'hypersphere'
    Attributes
    ----------
    coef_ : array-like of shape (n_features,)
        Coefficients of the regression model (mean of distribution).
    intercept_ : float
        Independent term in decision function. Set to 0.0 if
        `fit_intercept = False`.
    alpha_ : float
        Estimated precision of the noise. Not used for prediction.
    lambda_ : float
        Estimated epistemic precision of the weights.
    sigma_ : array-like of shape (n_features, n_features)
        Estimated epstemic variance-covariance matrix of the weights.
    misspecification_sigma_ : array-like of shape (n_features, n_features)
        Estimated misspecification variance-covariance matrix of the weights.
    scores_ : list
        If computed, value of the objective function (to be maximized).

    Notes
    -----
    The POPS algorithm extends Bayesian Ridge Regression for low-noise, misspecified regression problems, which can lead to improved performance in uncertainty prediction, suitable for high-dimensional settings.

    References
    ----------
    .. [1] Swinburne, T.D and Perez, D (2024). 
           Parameter uncertainties for imperfect surrogate models in the low-noise regime, arXiv:2402.01810v3
    """
    _parameter_constraints: dict = {
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "tol": [Interval(Real, 0, None, closed="neither")],
        "alpha_1": [Interval(Real, 0, None, closed="left")],
        "alpha_2": [Interval(Real, 0, None, closed="left")],
        "lambda_1": [Interval(Real, 0, None, closed="left")],
        "lambda_2": [Interval(Real, 0, None, closed="left")],
        "alpha_init": [None, Interval(Real, 0, None, closed="left")],
        "lambda_init": [None, Interval(Real, 0, None, closed="left")],
        "compute_score": ["boolean"],
        "fit_intercept": ["boolean"],
        "copy_X": ["boolean"],
        "verbose": ["verbose"],
        "resampling_method": [StrOptions({"uniform","sobol","latin","halton"})],
        "mode_threshold": [Interval(Real, 0, None, closed="neither")],
        "resample_density": [Interval(Real, 0, None, closed="neither")],
        "percentile_clipping": [Interval(Real, 0, 50., closed="both")],
        "leverage_percentile": [Interval(Real, 0.0, 100., closed="left")],
        "posterior": [StrOptions({"hypercube", "ensemble", "hypersphere"})],
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
        fit_intercept=False,
        copy_X=True,
        verbose=False,
        mode_threshold=1.0e-8,
        resample_density=1.0,
        resampling_method='uniform',
        percentile_clipping=0.0,
        leverage_percentile=50.0,
        posterior='hypercube',
    ):
        super().__init__(
            max_iter=max_iter,
            tol=tol,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
            alpha_init=alpha_init,
            lambda_init=lambda_init,
            compute_score=compute_score,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
            verbose=verbose,
        )
        self.posterior = posterior
        self.mode_threshold = mode_threshold
        self.fit_intercept_flag = False
        self.resample_density = resample_density
        self.resampling_method = resampling_method
        self.percentile_clipping = percentile_clipping
        self.leverage_percentile = leverage_percentile
        self.pointwise_correction = None
        self._validate_params()

        if self.fit_intercept:
            print("Warning: fit_intercept is set to False for POPS regression. A constant feature will be added to the design matrix.")
            self.fit_intercept_flag = True
            self.fit_intercept = False
        
    
    
    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        """Fit the model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values. Will be cast to X's dtype if necessary.

        sample_weight : ndarray of shape (n_samples,), default=None
            Individual weights for each sample.

            .. versionadded:: 0.20
               parameter *sample_weight* support to BayesianRidge.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        if self.fit_intercept_flag:
            # add a constant feature
            X = np.hstack([X, np.ones((X.shape[0], 1))])

        X, y = validate_data(self,
            X, y, dtype=[np.float64, np.float32])
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
        # is zero
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

        self.errors = (y - X @ self.coef_)
        
        # Pointwise corrections and leverage
        self.pointwise_correction = np.dot(X,scaled_sigma_)
        self.leverage_scores = np.sum(self.pointwise_correction * X,axis=1)
        self.pointwise_correction *= (self.errors / self.leverage_scores)[:,None]
        # NEW 02/2025: only consider *upper* 50% of leverage corrections 
        # to accelerate hypercube fitting and avoid "zero leverage" errors
        self.leverage_mask = \
            self.leverage_scores >= np.percentile(self.leverage_scores,self.leverage_percentile)
        
        self.posterior_samples, self.misspecification_sigma_ = self.build_posterior()

        self.sigma_ = (1.0 / alpha_) * scaled_sigma_
        
        self._set_intercept(X_offset_, y_offset_, X_scale_)

        return self

    def build_posterior(self):
        assert not self.pointwise_correction is None
        pc = self.pointwise_correction[self.leverage_mask]
        if self.posterior == 'ensemble':
            sigma = pc.T@pc / pc.shape[0]
            return pc.T, sigma
        
        elif self.posterior == 'hypercube':
            self.hypercube_support, self.hypercube_bounds = \
                self._hypercube_fit(pc)
            return self._sample_hypercube() # samples, sigma
        
        elif self.posterior == 'hypersphere':
            AD = pc / (self.errors / self.leverage_scores)[self.leverage_mask,None]
            # Fisher hypersphere
            fisher_inv_sqrt = linalg.sqrtm(AD.T@AD)/AD.shape[1] # this is the sqrt of the inverse fisher matrix
            fisher_radii = np.linalg.norm(linalg.solve(fisher_inv_sqrt,pc.T),axis=0)
            fisher_radius = np.percentile(fisher_radii,100-self.percentile_clipping)
            n_resample = int(self.resample_density*self.leverage_scores.size)

            hypersphere_samples = np.random.normal(size=(n_resample,AD.shape[1]))
            hypersphere_samples /= np.linalg.norm(hypersphere_samples,axis=1)[:,None]
            r = np.random.uniform(0.01,1.0,hypersphere_samples.shape[0])
            hypersphere_samples *= np.exp(np.log(r))[:,None] 
            hypersphere_samples *= fisher_radius
            hypersphere_samples = (hypersphere_samples@fisher_inv_sqrt)

            sigma = hypersphere_samples.T@hypersphere_samples / hypersphere_samples.shape[0]
            return hypersphere_samples.T, sigma
    

    def _hypercube_fit(self,pointwise_correction):
        """
        Fit a hypercube to the pointwise corrections.

        This method calculates the principal components of the pointwise corrections
        and determines the bounding box (hypercube) in the space of these components.

        Parameters:
        -----------
        pointwise_correction : numpy.ndarray
            Array of pointwise corrections, shape (n_samples, n_features).

        Returns:
        --------
        projections : numpy.ndarray
            The principal component vectors that define the hypercube space.
        bounds : numpy.ndarray
            The min and max bounds of the hypercube along each principal component.

        Notes:
        ------
        The method performs the following steps:
        1. Compute the eigendecomposition of the covariance matrix of pointwise corrections.
        2. Select principal components based on the mode_threshold.
        3. Project the pointwise corrections onto these components.
        4. Determine the bounding box (hypercube) in this projected space.

        The resulting hypercube represents the uncertainty in the parameter estimates,
        which can be used for subsequent resampling and uncertainty quantification.
        """
        
        e_values, e_vectors = eigh(pointwise_correction.T @pointwise_correction)
        
        mask = e_values > self.mode_threshold * e_values.max()
        e_vectors = e_vectors[:,mask]
        e_values = e_values[mask]
        
        projections = e_vectors.copy()
        projected = pointwise_correction @ projections
        bounds = [np.percentile(projected,self.percentile_clipping,axis=0)]
        bounds += [np.percentile(projected,100.-self.percentile_clipping,axis=0)]
        
        return projections, bounds
    
    def _sample_hypercube(self,size=None,resampling_method=None):
        """
        Resample points from the hypercube.

        This method generates new samples from the hypercube defined by the
        bounding box of the pointwise corrections. The sampling is uniform
        within the hypercube.

        Parameters:
        -----------
        size : int, optional
            The number of samples to generate. If None, the number of samples
            is determined by self.resample_density * self.leverage_scores.size.

        Returns:
        --------
        numpy.ndarray
            An array of shape (n_features, n_samples) containing the resampled
            points in the feature space.

        Notes:
        ------
        The resampling process involves the following steps:
        1. Generate uniform random numbers between 0 and 1.
        2. Scale these numbers to the range of the hypercube bounds.
        3. Project the scaled points back to the original feature space using
           the hypercube support vectors.

        This method is used to generate new possible parameter values within
        the uncertainty bounds of the model, which can be used for uncertainty
        quantification in predictions.
        """
        if resampling_method is None:
            resampling_method = self.resampling_method
        
        # Validate resampling_method parameter
        valid_methods = ['latin', 'sobol', 'grid', 'halton', 'uniform']
        if resampling_method not in valid_methods:
            raise ValueError(f"Invalid resampling_method. Must be one of {valid_methods}")

        low = self.hypercube_bounds[0]
        high = self.hypercube_bounds[1]
        if size is None:
            n_resample = int(self.resample_density*self.leverage_scores.size)
        else:
            n_resample = size
        n_resample = max(n_resample,100)
        
        # Sobol sequence
        if resampling_method == 'latin':
            sampler = qmc.LatinHypercube(d=low.size)
            samples = sampler.random(n_resample).T 
        elif resampling_method == 'sobol':
            sampler = qmc.Sobol(d=low.size)
            n_resample = 2**int(np.log(n_resample)/np.log(2.0))
            samples = sampler.random(n_resample).T 
        elif resampling_method == 'grid':
            samples = np.linspace(0,1,n_resample).T
        elif resampling_method == 'halton':
            sampler = qmc.Halton(d=low.size)
            samples = sampler.random(n_resample).T 
        elif resampling_method == 'uniform':
            samples = np.random.uniform(size=(low.size,n_resample))
        samples = low[:,None] + (high-low)[:,None]*samples

        hypercube_samples = self.hypercube_support@samples
        hypercube_sigma = hypercube_samples@hypercube_samples.T
        hypercube_sigma /= hypercube_samples.shape[1]

        return hypercube_samples,hypercube_sigma
    


    def predict(self,X,return_std=False,return_bounds=False,return_epistemic_std=False):
        """
        Make predictions using the POPS model.

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            The input samples for prediction.
        return_std : bool, default=False
            If True, return the combined misspecification and epistemic uncertainties.
        return_bounds : bool, default=False
            If True, return the min and max bounds of the prediction.
        return_epistemic_std : bool, default=False
            If True, return the epistemic standard deviation.

        Returns:
        --------
        y_mean : array-like of shape (n_samples,)
            The predicted mean values.
        y_std : array-like of shape (n_samples,)
            The predicted standard deviation (uncertainty) for each prediction.
        y_max : array-like of shape (n_samples,), optional
            The upper bound of the prediction interval. Only returned if return_bounds is True.
        y_min : array-like of shape (n_samples,), optional
            The lower bound of the prediction interval. Only returned if return_bounds is True.
        """
        if self.fit_intercept_flag:
            X = np.hstack([X, np.ones((X.shape[0], 1))])
        
        y_mean = self._decision_function(X)
        
        res = [y_mean] # y_mean
        
        # Follows rather odd convention of sklearn
        if return_std or return_bounds or return_epistemic_std:
            # we do NOT include aleatoric error !
            y_epistemic_var = (np.dot(X, self.sigma_) * X).sum(axis=1)
            y_max = (X@self.posterior_samples).max(1) + y_mean 
            y_min = (X@self.posterior_samples).min(1) + y_mean 
            
            if return_std:
                # Combine misspecification and epistemic uncertainty
                y_misspecification_var = (np.dot(X, self.misspecification_sigma_) * X).sum(axis=1)
                res += [np.sqrt(y_misspecification_var+ y_epistemic_var)] # y_std
            if return_bounds:
                res += [y_max,y_min]
            if return_epistemic_std:
                res += [y_epistemic_std] # y_epistemic_std
        
        return tuple(res)
