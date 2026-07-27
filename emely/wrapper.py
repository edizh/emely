from .normal import NormalMLE
from .poisson import PoissonMLE
from .laplace import LaplaceMLE
from .folded_normal import FoldedNormalMLE
from .rayleigh import RayleighMLE
from .rice import RiceMLE


def curve_fit(
    f,
    xdata,
    ydata,
    p0=None,
    bounds=None,
    sigma=None,
    absolute_sigma=False,
    method="nelder-mead",
    noise="normal",
    **optimizer_kwargs,
):
    """
    This function provides a scipy.optimize.curve_fit-like interface for maximum likelihood
    estimation (MLE) fitting with support for different noise distributions. It wraps the
    BaseMLE classes to provide a familiar interface to the scipy.optimize.curve_fit function.

    The function accepts xdata in the same format as scipy.optimize.curve_fit:
    - For single-variable models: 1D array with shape (num_data,)
    - For multi-variable models: 2D array with shape (num_vars, num_data)

    Parameters
    ----------
    f : callable
        The model function, f(x, ...). It must take the independent variable as the first
        argument and the parameters to fit as separate remaining arguments.
    xdata : array_like
        The independent variable. For single-variable models, can be 1D with shape (num_data,).
        For multi-variable models, must be 2D with shape (num_vars, num_data).
        Compatible with scipy.optimize.curve_fit format.
    ydata : array_like
        The dependent data. Shape (num_data,).
    p0 : array_like, optional
        Initial guess for the parameters. Shape (num_params,). Default is None.
    bounds : 2-tuple of array_like or scipy.optimize.Bounds, optional
        Bounds for the parameters as (lower_bounds, upper_bounds). Each side is either
        a scalar, which is broadcast to all parameters, or an array_like with shape
        (num_params,). Use None for no bound. Default is None.
    sigma : array_like, optional
        Uncertainties (standard deviation) in ydata with shape (num_data,).
        May be used depending on the noise distribution. The array is never modified.
    absolute_sigma : bool, optional
        If True, sigma is the absolute standard deviation of the noise.
        If False, the absolute standard deviation is estimated from the data.
        Default is False.
    method : str, optional
        Optimization method for scipy.optimize.minimize. Default is "nelder-mead".
        This parameter takes precedence over any `method` specified in `optimizer_kwargs`.
    noise : {"normal", "poisson", "laplace", "folded-normal", "rayleigh", "rice"}, optional
        Noise type for maximum likelihood estimation. Default is "normal".
        - "normal": Assumes a normal noise distribution
        - "poisson": Assumes a Poisson noise distribution
        - "laplace": Assumes a Laplace noise distribution
        - "folded-normal": Assumes a folded normal noise distribution
        - "rayleigh": Assumes a Rayleigh noise distribution
        - "rice": Assumes a Rice noise distribution
    **optimizer_kwargs
        Additional keyword arguments passed to scipy.optimize.minimize.
        Note: The `method` parameter will override any `method` key in `optimizer_kwargs`.

    Returns
    -------
    popt : ndarray
        Optimal parameter values. Shape (num_params,).
    pcov : ndarray
        Estimated covariance matrix. Shape (num_params, num_params).
        Filled with infinity if the parameters are not constrained by the data.

    Raises
    ------
    ValueError
        If the noise type is invalid, if the data is empty or contains NaNs or infs,
        or if the number of parameters cannot be determined from p0 and bounds.
    RuntimeError
        If the optimizer did not converge.
    """

    model = f
    x_data = xdata
    y_data = ydata
    params_init = p0
    param_bounds = bounds
    sigma_y = sigma
    is_sigma_y_absolute = absolute_sigma
    optimizer_kwargs["method"] = method

    if noise == "normal":
        MLE = NormalMLE
    elif noise == "poisson":
        MLE = PoissonMLE
    elif noise == "laplace":
        MLE = LaplaceMLE
    elif noise == "folded-normal":
        MLE = FoldedNormalMLE
    elif noise == "rayleigh":
        MLE = RayleighMLE
    elif noise == "rice":
        MLE = RiceMLE
    else:
        raise ValueError(f'Invalid noise type "{noise}".')

    estimator = MLE(model, **optimizer_kwargs)

    params, params_cov = estimator.fit(
        x_data,
        y_data,
        params_init,
        param_bounds,
        sigma_y,
        is_sigma_y_absolute,
    )

    popt = params
    pcov = params_cov

    return popt, pcov
