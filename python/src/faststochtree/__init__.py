import numpy as np

from ._faststochtree import BARTConfig
from ._faststochtree import BARTModel as _BARTModel
from ._faststochtree import fit_bart as _fit_bart
from ._faststochtree import fit_xbart as _fit_xbart


class BARTModel:
    def __init__(self, _model: _BARTModel, y_mean: float, y_sd: float):
        self._model = _model
        self._y_mean = float(y_mean)
        self._y_sd   = float(y_sd)

    @property
    def n_samples(self) -> int:
        return self._model.n_samples

    @property
    def n_test(self) -> int:
        return self._model.n_test

    @property
    def test_samples(self) -> np.ndarray:
        return self._model.test_samples * self._y_sd + self._y_mean

    @property
    def sigma2_samples(self) -> np.ndarray:
        return self._model.sigma2_samples * self._y_sd ** 2

    def predict(self, X_new: np.ndarray) -> np.ndarray:
        return self._model.predict(X_new) * self._y_sd + self._y_mean


def _scale_y(y: np.ndarray):
    y = np.asarray(y, dtype=np.float32)
    y_mean = float(y.mean())
    y_sd   = float(y.std())
    if y_sd == 0.0:
        y_sd = 1.0
    return (y - y_mean) / y_sd, y_mean, y_sd


def fit_bart(X, y, X_test, n_burnin: int = 200, n_samples: int = 1000,
             seed: int = 42, config: BARTConfig = None) -> BARTModel:
    if config is None:
        config = BARTConfig()
    y_scaled, y_mean, y_sd = _scale_y(y)
    if config.leaf_prior_var < 0.0:
      config.leaf_prior_var = 1.0 / config.num_trees
    return BARTModel(
        _fit_bart(X, y_scaled, X_test, n_burnin, n_samples, seed, config),
        y_mean, y_sd)


def fit_xbart(X, y, X_test, n_burnin: int = 15, n_samples: int = 25,
              seed: int = 42, config: BARTConfig = None) -> BARTModel:
    if config is None:
        config = BARTConfig()
    y_scaled, y_mean, y_sd = _scale_y(y)
    if config.leaf_prior_var < 0.0:
      config.leaf_prior_var = 1.0 / config.num_trees
    return BARTModel(
        _fit_xbart(X, y_scaled, X_test, n_burnin, n_samples, seed, config),
        y_mean, y_sd)


__all__ = ["BARTConfig", "BARTModel", "fit_bart", "fit_xbart"]
