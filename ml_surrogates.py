#ml_surrogates.py

# PyE+ - A Python-EnergyPlus Optimization Framework
# Copyright (c) 2025 Dr. Mubashir Hussain Wani
# Licensed under the MIT License. See LICENSE file in the project root for full license text.

"""
Surrogate model abstractions for surrogate-assisted NSGA-II.
Implements RandomForest, GaussianProcess, and MLPRegressor surrogates.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neural_network import MLPRegressor


class SurrogateModelBase:
    """
    Base class interface for surrogate models.
    """
    def fit(self, X, y):
        """Train the surrogate on input data X (n_samples, n_features) and targets y (n_samples, n_targets)."""
        raise NotImplementedError

    def predict(self, X):
        """Predict objective values for each row in X."""
        raise NotImplementedError


class RandomForestSurrogate(SurrogateModelBase):
    def __init__(self, n_estimators=100, **kwargs):
        self.model = RandomForestRegressor(n_estimators=n_estimators, **kwargs)

    def fit(self, X, y):
        # y shape: (n_samples, n_targets) -> multi-output regression
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


class GaussianProcessSurrogate(SurrogateModelBase):
    def __init__(self, kernel=None, **kwargs):
        self.model = GaussianProcessRegressor(kernel=kernel, **kwargs)

    def fit(self, X, y):
        # For multi-output, fit separate GPs or wrap accordingly
        # Here we train one GP per target dimension
        self.models_ = []
        for i in range(y.shape[1]):
            #gp = GaussianProcessRegressor(kernel=self.model.kernel, **self.model.get_params())
            params = self.model.get_params()
            params.pop('kernel', None)
            gp = GaussianProcessRegressor(kernel=self.model.kernel, **params)
            gp.fit(X, y[:, i])
            self.models_.append(gp)

    def predict(self, X):
        preds = [gp.predict(X) for gp in self.models_]
        # stack into shape (n_samples, n_targets)
        import numpy as np
        return np.stack(preds, axis=1)


class MLPSurrogate(SurrogateModelBase):
    def __init__(self, hidden_layer_sizes=(64, 64), max_iter=500, **kwargs):
        self.model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                   max_iter=max_iter, **kwargs)

    def fit(self, X, y):
        # Flatten y to shape (n_samples, n_targets) for MLP
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)


# Factory for dynamic creation
def make_surrogate(model_name, **kwargs):
    if model_name == 'RandomForest':
        return RandomForestSurrogate(**kwargs)
    elif model_name == 'GaussianProcess':
        return GaussianProcessSurrogate(**kwargs)
    elif model_name == 'MLPRegressor':
        return MLPSurrogate(**kwargs)
    else:
        raise ValueError(f"Unknown surrogate type: {model_name}")
