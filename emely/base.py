import numpy as np
from scipy.optimize import minimize, differential_evolution, Bounds
from scipy.stats import median_abs_deviation
import numdifftools as nd
from abc import ABC, abstractmethod


class BaseMLE(ABC):
    """
    Base class for maximum likelihood estimation.

    This class provides common functionality for fitting models with different
    noise distributions (Poisson, Normal, Laplace, Folded Normal, Rayleigh, Rice).
    It handles parameter estimation, covariance matrix calculation via the Fisher
    Information Matrix, model prediction with uncertainty, and model selection
    criteria (AIC and BIC).
    """

    def __init__(
        self,
        model,
        verbose=False,
        **optimizer_kwargs,
    ):
        """
        Initialize the BaseMLE.

        Parameters
        ----------
        model : callable
            The model function, f(x_data, *params).
        verbose : bool, optional
            If True, print the optimization results. Default is False.
        **optimizer_kwargs
            Keyword arguments passed to scipy.optimize.minimize. Default values:
            - method : str, default "nelder-mead"
                Optimization method to use.
            - tol : float, default 1e-9
                Tolerance for termination. Its meaning depends on the method:
                scipy.optimize.minimize maps it to the absolute xatol and fatol for
                "nelder-mead", but to a relative ftol for e.g. "l-bfgs-b". A large
                negative log-likelihood can therefore put fatol out of reach.
            Any additional keyword arguments are also passed through to
            scipy.optimize.minimize. User-provided values override the defaults.
        """
        self._model = self._wrap_model(model)
        self._params_init = None
        self._param_bounds = None
        self._params = None
        self._param_covs = None
        self._sigma_y = None
        self._is_sigma_y_absolute = None
        self._akaike_info = None
        self._bayesian_info = None
        self.verbose = verbose

        default_optimizer_kwargs = {
            "method": "nelder-mead",
            "tol": 1e-9,
        }
        self.optimizer_kwargs = {**default_optimizer_kwargs, **optimizer_kwargs}

    @property
    def model(self):
        """The wrapped model function."""
        return self._model

    @property
    def params_init(self):
        """Initial parameter guess used for optimization."""
        return self._params_init

    @property
    def param_bounds(self):
        """Parameter bounds used for optimization."""
        return self._param_bounds

    @property
    def params(self):
        """Optimal parameter values from the fit."""
        return self._params

    @property
    def param_covs(self):
        """Parameter covariance matrix from the fit."""
        return self._param_covs

    @property
    def sigma_y(self):
        """Uncertainties (standard deviation) in y_data."""
        return self._sigma_y

    @property
    def is_sigma_y_absolute(self):
        """If True, sigma_y is the absolute standard deviation of the noise."""
        return self._is_sigma_y_absolute

    @property
    def akaike_info(self):
        """
        Akaike Information Criterion (AIC) of the fit.

        The AIC measures the relative quality of a statistical model by balancing
        model fit (negative log-likelihood) against model complexity (number of parameters).
        It is calculated as:

            aic = 2 * nll + 2 * num_params

        where NLL is the negative log-likelihood.
        A smaller AIC indicates a better model. The AIC is computed automatically
        after calling fit().

        Returns
        -------
        float
            Akaike Information Criterion of the fit.
        """
        return self._akaike_info

    @property
    def bayesian_info(self):
        """
        Bayesian Information Criterion (BIC) of the fit.

        The BIC measures the relative quality of a statistical model by balancing
        model fit (negative log-likelihood) against model complexity (number of parameters).
        It is calculated as:

            bic = 2 * nll + num_params * log(num_data)

        where NLL is the negative log-likelihood.
        A smaller BIC indicates a better model. The BIC is computed automatically
        after calling fit().

        Returns
        -------
        float
            Bayesian Information Criterion of the fit.
        """
        return self._bayesian_info

    @staticmethod
    def _wrap_model(model):
        def model_wrapped(x_data, *params):
            y = np.asarray(model(x_data, *params))
            y = np.squeeze(y)

            return y

        return model_wrapped

    @staticmethod
    def _standardize_param_bounds(param_bounds, num_params):
        """
        Standardize the parameter bounds to a list of (lower, upper) tuples.

        Scalar bounds are broadcast to all parameters, `None` is replaced by the
        corresponding infinite bound, and `scipy.optimize.Bounds` instances are
        accepted. This mirrors the bounds handling of scipy.optimize.curve_fit.

        Parameters
        ----------
        param_bounds : array_like or scipy.optimize.Bounds
            Bounds for the parameters as (lower_bounds, upper_bounds). Each side is
            either a scalar, which is broadcast to all parameters, or an array_like
            with shape (num_params,). Use None for no bound.
        num_params : int or None
            Number of parameters, if already known. If None, it is inferred from the
            bounds, which is only possible if at least one side is not a scalar.

        Returns
        -------
        param_bounds : list or None
            Standardized parameter bounds. List of (lower, upper) tuples with length
            num_params. None if num_params could not be determined.
        num_params : int or None
            Number of parameters. None if it could not be determined.

        Raises
        ------
        ValueError
            If the number of bounds is not compatible with the number of parameters.
        """
        if isinstance(param_bounds, Bounds):
            bound_sides = (param_bounds.lb, param_bounds.ub)
        else:
            bound_sides = tuple(param_bounds)

        if len(bound_sides) != 2:
            raise ValueError(
                "Parameter bounds must be given as (lower_bounds, upper_bounds)."
            )

        no_bound_values = (-np.inf, np.inf)
        standardized_sides = []

        for bounds, no_bound_value in zip(bound_sides, no_bound_values):
            if bounds is None:
                bounds = no_bound_value

            if np.ndim(bounds) == 0:
                bounds = np.asarray(bounds, dtype=float)
            else:
                bounds = np.asarray(
                    [no_bound_value if b is None else b for b in bounds], dtype=float
                )

            standardized_sides.append(bounds)

        lower_bounds, upper_bounds = standardized_sides

        if num_params is None:
            if lower_bounds.ndim > 0:
                num_params = lower_bounds.size
            elif upper_bounds.ndim > 0:
                num_params = upper_bounds.size

        if num_params is None:
            return None, None

        if lower_bounds.ndim == 0:
            lower_bounds = np.resize(lower_bounds, num_params)

        if upper_bounds.ndim == 0:
            upper_bounds = np.resize(upper_bounds, num_params)

        if lower_bounds.size != num_params or upper_bounds.size != num_params:
            raise ValueError(
                "The number of bounds is not compatible with the number of parameters."
            )

        param_bounds = list(zip(lower_bounds, upper_bounds))

        return param_bounds, num_params

    def _check_fit_args(
        self,
        x_data,
        y_data,
        params_init,
        param_bounds,
        sigma_y,
        is_sigma_y_absolute,
    ):
        """
        Check and normalize arguments for the fit method.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters. Shape (num_params,). Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds). Each side is
            either a scalar, which is broadcast to all parameters, or an array_like
            with shape (num_params,). Use None for no bound. Default is None.
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        x_data : ndarray
            Independent variable, converted to a float array. Shape matches input:
            (num_data,) for 1D input or (num_vars, num_data) for 2D input. Inputs that
            are not array_like are passed through as-is, since they are handed straight
            to the model.
        y_data : ndarray
            Dependent data. Shape (num_data,).
        params_init : list
            Initial parameter guess, one entry per fitted parameter, including the
            appended scale parameter where one is fitted. Entries are None where no
            guess was provided.
        param_bounds : list
            Standardized parameter bounds. One (lower, upper) tuple per entry of
            params_init.
        sigma_y : ndarray
            Uncertainties (standard deviation) in y_data with shape (num_data,). Always a
            fresh array, so the array provided by the caller is never modified.
        is_sigma_y_absolute : bool
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.

        Raises
        ------
        ValueError
            If the data is empty or contains NaNs or infs, if sigma_y is missing while
            is_sigma_y_absolute is True, or if the number of parameters cannot be
            determined from params_init and param_bounds.
        """
        y_data = np.asarray_chkfinite(y_data, dtype=float)

        if isinstance(x_data, (list, tuple, np.ndarray)):
            # x_data is handed straight to the model, so other types are passed through
            x_data = np.asarray_chkfinite(x_data, dtype=float)

        if np.size(y_data) == 0:
            raise ValueError("y_data must not be empty.")

        if sigma_y is None and is_sigma_y_absolute:
            raise ValueError("sigma_y must be provided if is_sigma_y_absolute=True")

        if sigma_y is None:
            sigma_y = np.ones_like(y_data, dtype=float)

        if np.ndim(sigma_y) == 0:
            sigma_y = np.full_like(y_data, sigma_y, dtype=float)

        sigma_y = np.array(sigma_y, dtype=float)

        if not is_sigma_y_absolute:
            sigma_y /= np.mean(sigma_y)

            dy_data = np.diff(y_data)
            weight = 1.4826 * median_abs_deviation(dy_data, scale=1) / np.sqrt(2)

            # scaling sigma_y by the noise scale conditions the objective without
            # moving its minimum, and is skipped when the scale cannot be estimated,
            # e.g. for constant or heavily quantized y_data
            if weight > 0:
                sigma_y *= weight

        num_params = None

        if params_init is not None:
            num_params = len(params_init)

        if param_bounds is not None:
            param_bounds, num_params = self._standardize_param_bounds(
                param_bounds, num_params
            )

        if num_params is None:
            raise ValueError(
                "Either initial parameters or parameter bounds must be provided."
            )

        if params_init is None:
            if not np.all(np.isfinite(param_bounds)):
                raise ValueError(
                    "Finite parameter bounds must be provided if no initial parameters are provided."
                )
            params_init = [None] * num_params

        if param_bounds is None:
            param_bounds = [(None, None)] * num_params

        if not self.is_semi_analytical and not is_sigma_y_absolute:
            params_init = list(params_init) + [1.0]
            param_bounds = list(param_bounds) + [(1e-1, 1e1)]

        return (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma_y,
            is_sigma_y_absolute,
        )

    def fit(
        self,
        x_data,
        y_data,
        params_init=None,
        param_bounds=None,
        sigma_y=None,
        is_sigma_y_absolute=False,
    ):
        """
        Perform maximum likelihood estimation fit.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data, nominally f(x_data, *params) with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters. Shape (num_params,). Default is None.
            If None, parameters are initialized using stochastic search (differential_evolution).
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds). Each side is
            either a scalar, which is broadcast to all parameters, or an array_like
            with shape (num_params,). Use None for no bound. Default is None.
        sigma_y : array_like, optional
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution. The array is never
            modified.
        is_sigma_y_absolute : bool, optional
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.
            Default is False.

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
            Filled with infinity if the parameters are not constrained by the data.

        Raises
        ------
        ValueError
            If the data is empty or contains NaNs or infs, if sigma_y is missing while
            is_sigma_y_absolute is True, or if the number of parameters cannot be
            determined from params_init and param_bounds.
        RuntimeError
            If the optimizer did not converge.
        """
        (
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma_y,
            is_sigma_y_absolute,
        ) = self._check_fit_args(
            x_data,
            y_data,
            params_init,
            param_bounds,
            sigma_y,
            is_sigma_y_absolute,
        )

        self._params_init = params_init
        self._param_bounds = param_bounds
        self._sigma_y = sigma_y
        self._is_sigma_y_absolute = is_sigma_y_absolute

        has_fitted_weight = not self.is_semi_analytical and not is_sigma_y_absolute

        self._params = self._estimate_parameters(x_data, y_data)

        self._sigma_y = self._estimate_absolute_sigma_y(x_data, y_data)
        self._is_sigma_y_absolute = True

        if has_fitted_weight:
            self._params = self._params[:-1]

        self._param_covs = self._estimate_covariances(x_data, y_data)

        self._akaike_info = self._estimate_akaike_info(
            x_data,
            y_data,
        )

        self._bayesian_info = self._estimate_bayesian_info(
            x_data,
            y_data,
        )

        return self.params, self.param_covs

    def predict(self, x_data):
        """
        Predict the model output for the given independent variable using the optimal parameters
        from the most recent fit() call.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).

        Returns
        -------
        y_pred : ndarray
            Predicted dependent data. Shape (num_data,).
        y_cov : ndarray
            Covariance matrix of the predicted dependent data. Shape (num_data, num_data).
            This represents the uncertainty in the predictions due to parameter uncertainty.
        """

        y_pred = self._model(x_data, *self.params)

        model = lambda params: self._model(x_data, *params)

        eps = np.finfo(float).eps ** (1 / 3)
        steps = eps * np.abs(self.params)
        steps = np.maximum(steps, np.finfo(float).eps)

        J = nd.Jacobian(model, method="complex", step=steps)(self.params)

        y_cov = J @ self.param_covs @ J.T

        return y_pred, y_cov

    def _estimate_parameters(
        self,
        x_data,
        y_data,
    ):
        """
        Estimate the model parameters using the maximum likelihood estimation.

        If params_init is None, parameters are first initialized using stochastic search
        (differential_evolution) before refinement with the standard optimizer.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        params : ndarray
            Optimal parameter values. Shape (num_params,).

        Raises
        ------
        RuntimeError
            If the optimizer did not converge, either because the iteration budget was
            too small, which options={"maxiter": ..., "maxfev": ...} resolves, or
            because tol is out of reach for the scale of the negative log-likelihood.
        """
        sigma_y = self._sigma_y
        is_sigma_y_absolute = self._is_sigma_y_absolute

        objective = lambda params: self._negative_log_likelihood(
            x_data, y_data, params, sigma_y, is_sigma_y_absolute
        )

        is_free = [p_i is None for p_i in self._params_init]

        if np.any(is_free):
            if self.verbose:
                print("Estimating initial parameters...")

            result = differential_evolution(
                objective,
                bounds=self._param_bounds,
                tol=self.optimizer_kwargs["tol"],
                polish=False,
            )

            if self.verbose:
                print("Initial parameters:", result.x)
                print("Success:", result.success)
                print("Iterations:", result.nit)
                print("Function calls:", result.nfev)
                print("Message:", result.message)
                print("--------------------------------")

            self._params_init = result.x

        if self.verbose:
            print("Estimating optimal parameters...")

        result = minimize(
            objective,
            x0=self._params_init,
            bounds=self._param_bounds,
            **self.optimizer_kwargs,
        )

        if self.verbose:
            print("Optimal parameters:", result.x)
            print("Success:", result.success)
            print("Iterations:", result.nit)
            print("Function calls:", result.nfev)
            print("Message:", result.message)
            print("--------------------------------")

        if not result.success:
            raise RuntimeError(f"Optimal parameters not found: {result.message}")

        params = result.x

        return params

    def _estimate_covariances(
        self,
        x_data,
        y_data,
    ):
        """
        Calculate the covariance matrix using the Cramér-Rao bound.

        If the Fisher information matrix is singular, the parameters are not
        constrained by the data and the covariance matrix is filled with infinity
        rather than being pseudo-inverted, since a pseudo-inverse would report a
        vanishing uncertainty for an unconstrained parameter.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        param_covs : ndarray
            Estimated covariance matrix. Shape (num_params, num_params).
            Filled with infinity if the Fisher information matrix is singular.
        """
        FIM = self._fisher_information_matrix(x_data, y_data)

        num_params = len(self.params)

        param_covs = np.full((num_params, num_params), np.inf)

        is_invertible = (
            np.shape(FIM) == (num_params, num_params)
            and np.all(np.isfinite(FIM))
            and np.linalg.matrix_rank(FIM) == num_params
        )

        if is_invertible:
            param_covs = np.linalg.inv(FIM)

        return param_covs

    def _estimate_akaike_info(
        self,
        x_data,
        y_data,
    ):
        """
        Calculate the Akaike Information Criterion (AIC).

        The AIC measures the relative quality of a statistical model by balancing
        model fit (negative log-likelihood) against model complexity (number of parameters).
        It is defined as:

            AIC = 2 * NLL + 2 * k

        where NLL is the negative log-likelihood and k is the number of parameters.
        A smaller AIC indicates a better model.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        aic : float
            Value of the Akaike Information Criterion.
        """
        sigma_y = self._sigma_y
        is_sigma_y_absolute = self._is_sigma_y_absolute
        params = self.params

        num_params = len(params)

        aic = (
            2
            * self._negative_log_likelihood(
                x_data, y_data, params, sigma_y, is_sigma_y_absolute
            )
            + 2 * num_params
        )
        return aic

    def _estimate_bayesian_info(
        self,
        x_data,
        y_data,
    ):
        """
        Calculate the Bayesian Information Criterion (BIC).

        The BIC measures the relative quality of a statistical model by balancing
        model fit (negative log-likelihood) against model complexity (number of parameters).
        It is defined as:

            bic = 2 * nll + num_params * log(num_data)

        where NLL is the negative log-likelihood.
        A smaller BIC indicates a better model.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        bic : float
            Value of the Bayesian Information Criterion.
        """
        sigma_y = self._sigma_y
        is_sigma_y_absolute = self._is_sigma_y_absolute
        params = self.params

        num_data = np.size(y_data)
        num_params = len(params)

        bic = 2 * self._negative_log_likelihood(
            x_data, y_data, params, sigma_y, is_sigma_y_absolute
        ) + num_params * np.log(num_data)

        return bic

    def _fisher_information_matrix(self, x_data, y_data):
        """
        Calculate the Fisher information matrix.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        FIM : ndarray
            Fisher information matrix. Shape (num_params, num_params).
        """
        sigma_y = self._sigma_y
        is_sigma_y_absolute = self._is_sigma_y_absolute
        params = self.params

        if self.is_semi_analytical:
            scale_squared = self._scale_squared
            S_sq_inv = np.diag(1 / scale_squared)

            model = lambda params: self._model(x_data, *params)

            eps = np.finfo(float).eps ** (1 / 3)
            steps = eps * np.abs(params)
            steps = np.maximum(steps, np.finfo(float).eps)

            # J = nd.Jacobian(model, method="complex", step=steps)(params)
            J = nd.Jacobian(model, method="central", step=steps)(params)

            FIM = J.T @ S_sq_inv @ J
        else:
            nll = lambda params: self._negative_log_likelihood(
                x_data, y_data, params, sigma_y, is_sigma_y_absolute
            )

            eps = np.finfo(float).eps ** (1 / 3)
            steps = eps * np.abs(params)
            steps = np.maximum(steps, np.finfo(float).eps)

            # logpdf doesn't accept complex numbers, so use central difference method
            H = nd.Hessian(nll, method="central", step=steps)(params)

            FIM = H

        return FIM

    @abstractmethod
    def _negative_log_likelihood(
        self, x_data, y_data, params, sigma_y, is_sigma_y_absolute
    ):
        """
        Calculate the negative log-likelihood.

        This method must be implemented by the subclass to define the specific
        negative log-likelihood for their noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).
        params : array_like
            Parameter values. Shape (num_params,).
        sigma_y : array_like
            Uncertainties (standard deviation) in y_data with shape (num_data,).
            May be used depending on the noise distribution.
        is_sigma_y_absolute : bool
            If True, sigma_y is the absolute standard deviation of the noise.
            If False, the absolute standard deviation is estimated from the data.

        Returns
        -------
        nll : float
            Value of the negative log-likelihood.
        """
        pass

    @abstractmethod
    def _estimate_absolute_sigma_y(self, x_data, y_data):
        """
        Estimate the absolute standard deviation of the noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable. For single-variable models, can be 1D with shape (num_data,).
            For multi-variable models, must be 2D with shape (num_vars, num_data).
        y_data : array_like
            The dependent data with shape (num_data,).

        Returns
        -------
        ndarray
            The estimated absolute standard deviation of the noise. Shape (num_data,).
        """
        pass

    @property
    @abstractmethod
    def _scale_squared(self):
        """
        The squared scale parameter of the noise distribution.

        Returns
        -------
        ndarray
            Squared scale parameter. Shape (num_data,).
        """
        pass

    @property
    @abstractmethod
    def is_semi_analytical(self):
        """
        Indicates whether the noise model supports a semi-analytical computation of the
        Fisher Information Matrix. If True, the FIM is evaluated using

            Jᵀ @ diag(1 / s^2) @ J,

        where J is the numerical Jacobian of the model. If False, the FIM is obtained
        via a numerical Hessian of the negative log-likelihood.

        Returns
        -------
        bool
            True if semi-analytical FIM computation is supported, False otherwise.
        """
        pass
