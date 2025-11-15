import numpy as np
from scipy.optimize import minimize
import numdifftools as nd
from abc import ABC, abstractmethod


class BaseMLE(ABC):
    """
    Base class for maximum likelihood estimation.

    This class provides common functionality for fitting models with different
    noise distributions (Poisson, Gaussian, etc.). Subclasses should implement the
    negative log-likelihood and Fisher information matrix to compute the covariance matrix.
    """

    def __init__(self, model):
        """
        Initialize the BaseMLE.

        Parameters
        ----------
        model : callable
            The model function, f(x_data, *params).
        """
        self.model = self._wrap_model(model)

    @staticmethod
    def _wrap_model(model):
        def model_wrapped(x_data, *params):
            y = np.asarray(model(x_data, *params))
            y = np.squeeze(y)

            return y

        return model_wrapped

    def fit(
        self,
        x_data,
        y_data,
        params_init=None,
        param_bounds=None,
        sigma=None,
        is_sigma_absolute=False,
        optimizer="nelder-mead",
        verbose=False,
        **kwargs,
    ):
        """
        Perform maximum likelihood estimation fit.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        y_data : array_like
            The dependent data, nominally f(x_data, *params) with shape (num_data,).
        params_init : array_like, optional
            Initial guess for the parameters with size num_params. Default is None.
        param_bounds : array_like, optional
            Bounds for the parameters as (lower_bounds, upper_bounds).
            Use None for no bound. Default is None.
        sigma : array_like, optional
            Uncertainties in y_data. May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.
        optimizer : str, optional
            Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
        verbose : bool, optional
            If True, print the optimization results. Default is False.
        **kwargs
            Additional keyword arguments passed to scipy.optimize.minimize.

        Returns
        -------
        params : array
            Optimal values for the num_params parameters so that the negative log-likelihood
            is minimized.
        params_cov : 2-D array
            The estimated covariance matrix of params of shape (num_params, num_params).
        """
        x_data = np.atleast_2d(x_data)

        if sigma is None and is_sigma_absolute:
            raise ValueError("sigma must be provided if is_sigma_absolute=True")
        if sigma is None:
            sigma = np.ones_like(y_data)
        if np.ndim(sigma) == 0:
            sigma = np.full_like(y_data, sigma)

        self.params = params_init
        param_bounds = list(zip(*param_bounds)) if param_bounds is not None else None

        result = minimize(
            lambda params: self.negative_log_likelihood(x_data, y_data, params, sigma),
            x0=params_init,
            bounds=param_bounds,
            method=optimizer,
            **kwargs,
        )

        if verbose:
            print("Optimal params:", result.x)
            print("Success:", result.success)
            print("Iterations:", result.nit)
            print("Function calls:", result.nfev)
            print("Message:", result.message)

        params = result.x
        params_cov = self.covariance_matrix(
            x_data, y_data, params, sigma, is_sigma_absolute
        )

        self.params = params
        self.params_cov = params_cov

        return params, params_cov

    def predict(self, x_data):
        """
        Predict the model output for the given independent variable using the optimal parameters.

        Parameters
        ----------
        x_data : array_like
            The independent variable.

        Returns
        -------
        y_pred : array_like
            The predicted model output.
        """

        return self.model(x_data, *self.params)

    @abstractmethod
    def negative_log_likelihood(self, x_data, y_data, params, sigma=None):
        """
        Calculate the negative log-likelihood for MLE estimation.

        This method must be implemented by the subclass to define the specific
        likelihood function for their noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable where the data is measured.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma : array_like, optional
            Uncertainties in y_data. May be used depending on the noise distribution.

        Returns
        -------
        nll : float
            Negative log-likelihood value.
        """
        pass

    @abstractmethod
    def fisher_information_matrix(
        self, x_data, y_data, params, sigma=None, is_sigma_absolute=False
    ):
        """
        Calculate the Fisher information matrix.

        This method must be implemented by the subclass to define the
        Fisher information matrix based on their specific noise distribution.

        Parameters
        ----------
        x_data : array_like
            The independent variable.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma : array_like, optional
            Uncertainties in y_data. May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        FIM : 2-D array
            Fisher information matrix of shape (num_params, num_params).
        """
        pass

    def covariance_matrix(
        self, x_data, y_data, params, sigma=None, is_sigma_absolute=False
    ):
        """
        Calculate the covariance matrix using the Fisher information matrix.

        Parameters
        ----------
        x_data : array_like
            The independent variable.
        y_data : array_like
            The dependent data.
        params : array_like
            Optimal parameter values.
        sigma : array_like, optional
            Uncertainties in y_data. May be used depending on the noise distribution.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        covariance : 2-D array
            The estimated covariance matrix.
        """
        FIM = self.fisher_information_matrix(
            x_data, y_data, params, sigma, is_sigma_absolute
        )
        try:
            covariance = np.linalg.inv(FIM)
        except np.linalg.LinAlgError:
            covariance = np.linalg.pinv(FIM)

        return covariance

    def jacobian(self, x_data, params):
        """
        Calculate the Jacobian matrix numerically.

        Parameters
        ----------
        x_data : array_like
            The independent variable with shape (num_vars, num_data).
        params : array_like
            Parameter values.

        Returns
        -------
        J : array
            Jacobian matrix of shape (num_data, num_params).
        """

        jacobian = nd.Jacobian(lambda p: self.model(x_data, *p), method="complex")
        J = jacobian(params)

        return J
