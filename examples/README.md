# Examples

The `examples/` folder contains demonstration notebooks showcasing the use of the Emely package for maximum likelihood estimation with different models and noise types:

## Example 1: Basic Gaussian fitting with Poisson noise
- **Purpose**: Demonstrates parameter estimation for a normalized Gaussian function with Poisson noise
- **Model**: 1D Gaussian function
- **Noise**: Poisson statistics
- **Features**: Uses `curve_fit` with `noise="poisson"` for maximum likelihood estimation

## Example 2: Gaussian fitting with Laplace noise
- **Purpose**: Demonstrates parameter estimation for a normalized Gaussian function with Laplace noise
- **Model**: 1D Gaussian function
- **Noise**: Laplace statistics
- **Features**: Uses `curve_fit` with `noise="laplace"` and parameter bounds for maximum likelihood estimation

## Example 3: Dynamic Gaussian with diffusion and folded normal noise
- **Purpose**: Demonstrates parameter estimation for a dynamic Gaussian function with time-dependent width (diffusion) and exponential decay, comparing Folded-Normal and Normal MLE estimators
- **Model**: 2D Gaussian function with diffusion (spatial and temporal dimensions)
- **Noise**: Folded normal statistics
- **Features**: Uses `FoldedNormalMLE` and `NormalMLE` classes to compare estimators, includes parameter uncertainty visualization and prediction plots

