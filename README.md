# Emely

**Emely** is a lightweight Python package for **maximum likelihood estimation (MLE)**–based parameter fitting.  
It provides a `scipy.optimize.curve_fit`-like interface built on top of `scipy.optimize.minimize`, with support for **Poisson** and **Gaussian** noise models.

---

## Features

- **MLE fitting** for Poisson and Gaussian noise  
- `emely.curve_fit` provides a `scipy.optimize.curve_fit`-like interface  
- Automatic **covariance matrix estimation** via the Fisher information matrix  
- Object-oriented API with `BaseMLE`, `GaussianMLE`, and `PoissonMLE` classes  

---

## Installation

Install from source using pip (requires Python 3.8+):

```bash
pip install .
```

For development, install in editable mode:

```bash
pip install -e .
```

Dependencies are managed via `pyproject.toml`.

## Quick Start

```python
# fit Gaussian using MLE for Poisson noise
import numpy as np
import matplotlib.pyplot as plt

from emely import curve_fit


# define the model
def gaussian(x, a, mu, sigma):
    return a / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# create the Gaussian data with Poisson noise
p = (100, 5, 1)
x_data = np.linspace(-10, 10, 1001)
y_data = np.random.poisson(gaussian(x_data, *p))

# fit using MLE for Poisson noise
p_opt, p_cov = curve_fit(
    gaussian,
    x_data,
    y_data,
    p0=(50, 10, 5),
    noise="poisson",
)

# show the fit
plt.plot(x_data, y_data, label="Measurement")
plt.plot(x_data, gaussian(x_data, *p_opt), label="Fit")

plt.grid()
plt.legend()
```

## API

The package provides both a function-based and class-based interface:

### Function-based API

```python
from emely import curve_fit

popt, pcov = curve_fit(f, xdata, ydata, p0=None, bounds=None, sigma=None, 
                       absolute_sigma=False, method="nelder-mead", 
                       noise="gaussian", **kwargs)
```

The `curve_fit` function mirrors `scipy.optimize.curve_fit` with an additional `noise` parameter:
- `noise="gaussian"`: Assumes Gaussian (normal) noise distribution (default)
- `noise="poisson"`: Assumes Poisson noise distribution

### Class-based API

For more advanced usage, you can use the MLE classes directly:

```python
from emely import GaussianMLE, PoissonMLE

# Create an estimator
estimator = PoissonMLE(model)

# Fit the data
params, params_cov = estimator.fit(x_data, y_data, params_init=p0, 
                                  param_bounds=bounds, sigma=sigma, 
                                  is_sigma_absolute=False, optimizer="nelder-mead")

# Predict using fitted parameters
y_pred = estimator.predict(x_new)
```

Available classes:
- `BaseMLE`: Abstract base class for MLE estimators
- `GaussianMLE`: MLE estimator for Gaussian noise
- `PoissonMLE`: MLE estimator for Poisson noise

## Why MLE for parameter estimation?

Maximum likelihood estimation (MLE) can correctly model Poisson- or Gaussian-distributed noise, leading to more accurate and unbiased parameter estimates as compared to least-squares fitting, which is only optimal for Gaussian-distributed noise.

## Examples

See the provided notebooks for detailed usage and comparisons:
- **example_1.ipynb:** Fitting a 1D Gaussian signal with Poisson noise
- **example_2.ipynb:** Fitting a 2D Gaussian signal with Poisson noise
- **example_3.ipynb:** Fitting a 2D Gaussian signal with temporally changing width and Poisson noise
- **example_4.ipynb:** Fitting 1D Gaussian signals with distinct Poisson noise to validate parameter covariances

These examples compare the accuracy of least-squares, Gaussian MLE, and Poisson MLE fits.

## License

MIT License © 2025

