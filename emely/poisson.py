import numpy as np
from .base import BaseMLE


class PoissonMLE(BaseMLE):
    """
    Maximum likelihood estimation for Poisson noise distribution.

    This class implements MLE fitting assuming the data follows a Poisson
    distribution.
    """

    def negative_log_likelihood(self, x_data, y_data, params, sigma=None):
        """
        Calculate the negative log-likelihood for Gaussian noise.

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
        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        nll = -np.sum(y_data * np.log(y_pred) - y_pred)

        return nll

    def fisher_information_matrix(
        self, x_data, y_data, params, sigma=None, is_sigma_absolute=False
    ):
        """
        Calculate the Fisher information matrix for Gaussian noise.

        Parameters
        ----------
        x_data : array_like
            The independent variable.
        y_data : array_like
            The dependent data.
        params : array_like
            Parameter values.
        sigma : array_like, optional
            Uncertainties in y_data.
        is_sigma_absolute : bool, optional
            If True, sigma is used for covariance matrix calculation.
            If False, covariances are calculated from residuals.
            Default is False.

        Returns
        -------
        FIM : 2-D array
            Fisher information matrix of shape (num_params, num_params).
        """
        y_pred = self.model(x_data, *params)
        y_pred = np.clip(y_pred, 1e-12, np.inf)

        w = 1 / y_pred
        W = np.diag(w)
        J = self.jacobian(x_data, params)

        FIM = J.T @ W @ J

        return FIM
