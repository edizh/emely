from .gaussian import GaussianMLE
from .poisson import PoissonMLE


def curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    bounds=None,
    sigma=None,
    absolute_sigma=False,
    method="nelder-mead",
    noise="gaussian",
    **kwargs,
):
    """
    This function provides a scipy.optimize.curve_fit-like interface for maximum likelihood
    estimation (MLE) fitting with support for different noise distributions. It wraps the
    BaseMLE classes to provide a familiar interface to the scipy.optimize.curve_fit function.

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent variable as the first
        argument and the parameters to fit as separate remaining arguments.
    xdata : array_like
        The independent variable with shape (num_vars, num_data).
    ydata : array_like
        The dependent data, nominally f(x_data, *params) with shape (num_data,).
    p0 : array_like, optional
        Initial guess for the parameters with size num_params. Default is None.
    bounds : 2-tuple of array_like, optional
        Bounds for the parameters as (lower_bounds, upper_bounds).
        Use None for no bound. Default is None.
    sigma : array_like, optional
        Uncertainties in y_data. May be used depending on the noise distribution.
    absolute_sigma : bool, optional
        If True, sigma is used for covariance matrix calculation.
        If False, covariances are calculated from residuals.
    method : str, optional
        Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
    noise : {"gaussian", "poisson"}, optional
        Noise type for maximum likelihood estimation. Default is "gaussian".
        - "gaussian": Assumes Gaussian (normal) noise distribution
        - "poisson": Assumes Poisson noise distribution
    **kwargs
        Additional keyword arguments passed to scipy.optimize.minimize.

    Returns
    -------
    popt : array
        Optimal values for the num_params parameters so that the negative log-likelihood
        is minimized.
    pcov : 2-D array
        The estimated covariance matrix of params of shape (num_params, num_params).
    """

    model = f
    x_data = xdata
    y_data = ydata
    params_init = p0
    param_bounds = bounds
    is_sigma_absolute = absolute_sigma
    optimizer = method
    verbose = False

    if noise == "gaussian":
        MLE = GaussianMLE
    elif noise == "poisson":
        MLE = PoissonMLE

    estimator = MLE(model)

    params, params_cov = estimator.fit(
        x_data,
        y_data,
        params_init,
        param_bounds,
        sigma,
        is_sigma_absolute,
        optimizer,
        verbose,
        **kwargs,
    )

    popt = params
    pcov = params_cov

    return popt, pcov
