import itertools
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_array


def prepare_data(x, y, kernels, x0=None):
    """Includes initial conditions in datasets for pipelines in which Volterra features are used.

    Parameters
    ----------
    x : array-like of shape (n_samples,)
        Vector of inputs.

    y : array-like of shape (n_samples,)
        Vector of outputs.

    kernels : tuple
              Volterra model kernels.

    x0 : array-like of shape (n_initial_conditions) or str, default=None
         Vector of initial conditions, i.e. [x[-1], x[-2], ...]. If x0 is None then initial conditions are extracted
         from x and y is shortened accordingly. If x0 is 'ones', then x[-1]=x[-2]=...=1

    Returns
    -------
    x : array-like of shape (n_out_samples,)
        Vector of modified inputs.
    y : array-like of shape (n_out_samples,)
        Vector of modified outputs.
    """

    model_memory_length = np.max(kernels)

    if x0 is None:
        offset = model_memory_length - 1
        y = y[offset:]
    elif isinstance(x0, str) and x0 == 'zeros':
        x0 = np.zeros([model_memory_length - 1, 1])
        x = np.concatenate([x0, x])
    else:
        assert len(x0) == model_memory_length - 1
        x = np.concatenate([x0, x])

    return x, y


class VolterraFeatures(TransformerMixin, BaseEstimator):
    """Generate polynomial-like features derived from the Volterra series expansion.

    Parameters
    ----------
    kernels : tuple, default=None
        Stores memory lengths (maximum time lags) of consecutive Volterra operators.
        For example, if kernels is (4, 2), then the memory length of the 1st order operator is equal to 4 (and
        corresponding features are x[t], x[t-1], x[t-2], x[t-3]) and the memory length of the 2nd order operator
        is equal to 2 (and corresponding features are x[t] ** 2, x[t-1] ** 2, x[t] * x[t-1]).

    include_bias : bool, default=True
        If True, then a constant element (bias) is included in generated features.
    """
    def __init__(self, kernels=None, include_bias=True):
        self.kernels = kernels
        self.include_bias = include_bias

    @staticmethod
    def _generate_indices(order, memory_length):
        return itertools.combinations_with_replacement(range(0, memory_length), order)

    @staticmethod
    def _volterra_function(indices, x, t):
        output = 1
        for i in indices:
            output *= x[t - i]
        return output

    def _generate_dictionary(self):
        functions = []

        if self.include_bias:
            functions.append(lambda x, t: 1)  # constant function

        order = 1
        for memory_length in self.kernels:
            indices = self._generate_indices(order, memory_length)
            for ind in indices:
                f = (lambda i: lambda x, t: self._volterra_function(i, x, t))(ind)
                functions.append(f)
            order += 1

        return functions

    def fit(self, X, y=None):
        """Compute number of output features and setup the transformer.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, 1)
            The input vector.
        y : None
            Ignored
        Returns
        -------
        self : object
        """
        X = check_array(X)

        if X.shape[1] != 1:
            raise ValueError('Input is required to be a column vector')

        self.memory_length_ = np.max(self.kernels)
        self.dictionary_ = self._generate_dictionary()
        self.n_features_ = len(self.dictionary_)

        return self

    def transform(self, X):
        """Transform data to Volterra features.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, 1    )
            The input vector.
        Returns
        -------
        X_transformed : array, shape (n_samples - memory_length + 1, n_features)
            The matrix of features, where n_features is the number of Volterra features generated from the combinations
            of past inputs.
        """
        check_is_fitted(self)
        X = check_array(X)

        if X.shape[1] != 1:
            raise ValueError('Input is required to be a column vector')

        n_samples = X.shape[0]
        n_output_samples = n_samples - self.memory_length_ + 1

        X_transformed = np.zeros([n_output_samples, self.n_features_])

        time_lags = list(range(self.memory_length_ - 1, n_samples))

        row_idx = 0
        for t in time_lags:
            row = np.hstack([f(X, t) for f in self.dictionary_])
            X_transformed[row_idx, :] = row
            row_idx += 1

        return X_transformed
