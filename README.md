# Emely

**Emely** is a lightweight Python package for **maximum likelihood estimation (MLE)**–based parameter fitting.  
It provides a `scipy.optimize.curve_fit`-like interface built on top of `scipy.optimize.minimize`, with support for **Poisson**, **Normal**, **Laplace**, **Folded Normal**, and **Rayleigh** noise models.

---

## Features

- **Parameter estimation** using MLE for Poisson, Normal, Laplace, Folded Normal, and Rayleigh noise 
- **Parameter error estimation** using the Fisher information matrix   
- `emely.curve_fit` provides a `scipy.optimize.curve_fit`-like interface  
- The underlying `BaseMLE` classes (`NormalMLE`, `PoissonMLE`, `LaplaceMLE`, `FoldedNormalMLE`, `RayleighMLE`) provide a modern object-oriented API

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
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from emely import curve_fit

# define plot style
blue = "#648fff"
red = "#dc267f"
black = "#000000"
plt.style.use("seaborn-v0_8")

# define the model
def gaussian(x, a, mu, sigma):
    return a / np.sqrt(2 * np.pi * sigma**2) * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

# create the Gaussian data with Poisson noise
p = (100, 5, 1)
x_data = np.linspace(0, 10, 51)
y_data = poisson.rvs(gaussian(x_data, *p))

# fit using MLE for Poisson noise
p0 = (50, 10, 5)
p_opt, p_cov = curve_fit(
    gaussian,
    x_data,
    y_data,
    p0=p0,
    noise="poisson",  # "normal", "poisson", "laplace", "folded-normal", "rayleigh"
)

# show the fit
plt.figure(figsize=(6, 2))

plt.scatter(x_data, y_data, label="Measurement", edgecolor=black, facecolor=blue)
plt.plot(x_data, gaussian(x_data, *p_opt), label="Fit", color=red)

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
```

## Why MLE for parameter estimation?

Maximum likelihood estimation (MLE) can correctly model different noise distributions, leading to more accurate and unbiased parameter estimates as compared to least-squares fitting, which is only optimal for normally-distributed noise.

## Examples

See the provided notebooks in the `examples/` folder for detailed usage:
- **example_1.ipynb:** Fitting a 1D Gaussian signal with Poisson noise
- **example_2.ipynb:** Fitting a 1D Gaussian signal with Laplace noise
- **example_3.ipynb:** Fitting a 2D dynamic Gaussian signal with folded normal-noise, comparing folded-normal and normal MLE estimators

For more details, see the [examples README](examples/README.md).

## License

MIT License © 2025

