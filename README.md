# Emely

A Python package for Maximum Likelihood Estimation (MLE) parameter fitting, designed for data with Poisson or Gaussian noise statistics.

## Features

- **MLE fitting** for Poisson and Gaussian noise models
- Can be used just like `scipy.optimize.curve_fit`
- Automatic covariance matrix estimation via Fisher information matrix
- Support for parameter bounds and custom optimization methods

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
# fit Gaussian using MLE for Poisson noise
import numpy as np
import matplotlib.pyplot as plt

from emely import mle_fit


# define the model
def gaussian(x, a, mu, sigma):
    return a / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


# create the Gaussian data with Poisson noise
p = (100, 5, 1)
x_data = np.linspace(-10, 10, 1001)
y_data = np.random.poisson(gaussian(x_data, *p))

# fit using MLE for Poisson noise
p_opt, p_cov = mle_fit(
    gaussian,
    x_data,
    y_data,
    p0=(50, 10, 5),
    noise_type="poisson",
)

# show the fit
plt.plot(x_data, y_data, label="Measurement")
plt.plot(x_data, gaussian(x_data, *p_opt), label="Fit")

plt.grid()
plt.legend()
```

## Why MLE for Poisson Data?

Least-squares optimization assumes Gaussian noise statistics. MLE can provide optimal estimates when the noise follows Poisson statistics.

## Documentation

See `example_1.ipynb` and `example_2.ipynb` for detailed usage examples with 1D and 2D Gaussian functions and benchmarking against least squares and Gaussian MLE algorithms.

