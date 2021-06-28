import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class EntropicDualAveraging(BaseEstimator):
    """The variant of the Dual Averaging algorithm based on the negative entropy function

    Parameters
    ----------
    l1_constraint : float, default=1.0
        Radius of an l1-norm ball.

    G : float, default=None
        This parameter scales the algorithm's step sizes. If set to None, then it is estimated according to
        the following formula: G = (l1_constraint + max(y)) ** 2 (see the documentation).

    learning_rate : {'constant', 'online', 'adaptive'}, default='adaptive'
        The strategy for choosing the algorithm's step sizes.
    """

    def __init__(self, l1_constraint=1.0, G=None, learning_rate='adaptive'):
        self.l1_constraint = l1_constraint
        self.G = G
        self.learning_rate = learning_rate

    def _validate_params(self):
        assert self.l1_constraint > 0

    def _estimate_G(self, y):
        return self.l1_constraint * (self.l1_constraint + np.max(y))

    @staticmethod
    def _map_parameters(parameters, R):
        D = int(len(parameters) / 2)

        transformed_parameters = np.zeros(D)

        for i in range(D):
            transformed_parameters[i] = R * (parameters[i] - parameters[i + D])

        return transformed_parameters

    @staticmethod
    def _compute_gradient(x, y, y_hat):
        return (y_hat - y) * x

    def fit(self, X, y):
        """Fits the model.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,) or (n_samples, n_outputs)
            The target values.
        Returns
        -------
        self : object
        """

        self._validate_params()
        X, y = check_X_y(X, y)

        n_samples, self.n_features_in_ = X.shape
        D = 2 * self.n_features_in_

        if self.G is None:
            G_sq = self._estimate_G(y) ** 2
        else:
            G_sq = self.G ** 2

        params_0 = np.ones(D) / D
        params_avg = params_0
        prev_iter_params = params_0

        eta = None
        gradient_sum = 0
        gradient_max_sq_sum = 0

        if self.learning_rate == 'constant':
            eta = self.G * self.l1_constraint / np.sqrt(n_samples)

        for i in range(n_samples):
            x = np.concatenate([X[i], -X[i]]) * self.l1_constraint
            y_hat = np.dot(prev_iter_params, x)
            gradient = self._compute_gradient(x, y[i], y_hat)

            if self.learning_rate == 'online':
                eta = self.G * self.l1_constraint / np.sqrt(n_samples)
            elif self.learning_rate == 'adaptive':
                eta = np.sqrt(np.log(D) / (G_sq + gradient_max_sq_sum))
                gradient_max_sq_sum += np.max(np.abs(gradient)) ** 2

            gradient_sum += gradient

            prev_iter_params = np.exp(-eta * gradient_sum)
            prev_iter_params /= np.linalg.norm(prev_iter_params, 1)

            params_avg = (prev_iter_params + params_avg * (i + 1)) / (i + 2)

        self.params_ = self._map_parameters(prev_iter_params, self.l1_constraint)

        return self

    def predict(self, X):
        """Predicts the target value.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
        """
        X = check_array(X)
        check_is_fitted(self)

        return X @ self.params_
