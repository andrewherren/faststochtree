import numpy as np

from ._faststochtree import BARTConfig
from ._faststochtree import BARTModel as _BARTModel
from ._faststochtree import fit_bart as _fit_bart
from ._faststochtree import fit_xbart as _fit_xbart
from ._faststochtree import fit_warmstart_bart as _fit_warmstart_bart
from ._faststochtree import BCFConfig
from ._faststochtree import BCFModel as _BCFModel
from ._faststochtree import fit_bcf as _fit_bcf
from ._faststochtree import fit_xbcf as _fit_xbcf
from ._faststochtree import fit_warmstart_bcf as _fit_warmstart_bcf


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


def fit_warmstart_bart(X, y, X_test,
                       n_gfr_burnin: int = 20, n_mcmc_burnin: int = 0,
                       n_samples: int = 200,
                       seed: int = 42,
                       keep_gfr_samples: bool = False,
                       config: BARTConfig = None) -> BARTModel:
    if config is None:
        config = BARTConfig()
    y_scaled, y_mean, y_sd = _scale_y(y)
    if config.leaf_prior_var < 0.0:
        config.leaf_prior_var = 1.0 / config.num_trees
    return BARTModel(
        _fit_warmstart_bart(X, y_scaled, X_test,
                             n_gfr_burnin, n_mcmc_burnin, n_samples,
                             seed, keep_gfr_samples, config),
        y_mean, y_sd)


class BCFModel:
    """Posterior samples from a fitted BCF model.

    tau predictions are CATE estimates τ(x) on the original y scale.
    mu predictions are the prognostic function μ(x, ê(x)) on the original y scale.
    """
    def __init__(self, _model: _BCFModel, y_mean: float, y_sd: float):
        self._model  = _model
        self._y_mean = float(y_mean)
        self._y_sd   = float(y_sd)

    @property
    def n_samples(self) -> int:
        return self._model.n_samples

    @property
    def n_test(self) -> int:
        return self._model.n_test

    @property
    def test_tau(self) -> np.ndarray:
        return self._model.test_tau * self._y_sd

    @property
    def test_mu(self) -> np.ndarray:
        return self._model.test_mu * self._y_sd + self._y_mean

    @property
    def sigma2_samples(self) -> np.ndarray:
        return self._model.sigma2_samples * self._y_sd ** 2

    def predict_tau(self, X_new: np.ndarray) -> np.ndarray:
        return self._model.predict_tau(np.asarray(X_new, dtype=np.float32)) * self._y_sd

    def predict_mu(self, X_new: np.ndarray, pi_hat_new: np.ndarray) -> np.ndarray:
        X_new   = np.asarray(X_new,     dtype=np.float32)
        pi_new  = np.asarray(pi_hat_new, dtype=np.float32).reshape(-1, 1)
        X_mu    = np.concatenate([X_new, pi_new], axis=1)
        return self._model.predict_mu(X_mu) * self._y_sd + self._y_mean


def fit_bcf(X, y, z, pi_hat, X_test, pi_hat_test,
            n_burnin: int = 200, n_samples: int = 1000,
            seed: int = 42, config: BCFConfig = None) -> BCFModel:
    if config is None:
        config = BCFConfig()
    if config.mu_config.leaf_prior_var < 0.0:
        config.mu_config.leaf_prior_var = 1.0 / config.mu_config.num_trees
    if config.tau_config.leaf_prior_var < 0.0:
        config.tau_config.leaf_prior_var = 1.0 / config.tau_config.num_trees
    y_scaled, y_mean, y_sd = _scale_y(y)
    z_f32       = np.asarray(z,           dtype=np.float32)
    pi_f32      = np.asarray(pi_hat,      dtype=np.float32)
    pi_test_f32 = np.asarray(pi_hat_test, dtype=np.float32)
    return BCFModel(
        _fit_bcf(X, y_scaled, z_f32, pi_f32, X_test, pi_test_f32,
                 n_burnin, n_samples, seed, config),
        y_mean, y_sd)


def fit_xbcf(X, y, z, pi_hat, X_test, pi_hat_test,
             n_burnin: int = 15, n_samples: int = 25,
             seed: int = 42, config: BCFConfig = None) -> BCFModel:
    if config is None:
        config = BCFConfig()
    if config.mu_config.leaf_prior_var < 0.0:
        config.mu_config.leaf_prior_var = 1.0 / config.mu_config.num_trees
    if config.tau_config.leaf_prior_var < 0.0:
        config.tau_config.leaf_prior_var = 1.0 / config.tau_config.num_trees
    y_scaled, y_mean, y_sd = _scale_y(y)
    z_f32       = np.asarray(z,           dtype=np.float32)
    pi_f32      = np.asarray(pi_hat,      dtype=np.float32)
    pi_test_f32 = np.asarray(pi_hat_test, dtype=np.float32)
    return BCFModel(
        _fit_xbcf(X, y_scaled, z_f32, pi_f32, X_test, pi_test_f32,
                  n_burnin, n_samples, seed, config),
        y_mean, y_sd)


def fit_warmstart_bcf(X, y, z, pi_hat, X_test, pi_hat_test,
                      n_gfr_burnin: int = 20, n_mcmc_burnin: int = 0,
                      n_samples: int = 200,
                      seed: int = 42,
                      keep_gfr_samples: bool = False,
                      config: BCFConfig = None) -> BCFModel:
    if config is None:
        config = BCFConfig()
    if config.mu_config.leaf_prior_var < 0.0:
        config.mu_config.leaf_prior_var = 1.0 / config.mu_config.num_trees
    if config.tau_config.leaf_prior_var < 0.0:
        config.tau_config.leaf_prior_var = 1.0 / config.tau_config.num_trees
    y_scaled, y_mean, y_sd = _scale_y(y)
    z_f32       = np.asarray(z,           dtype=np.float32)
    pi_f32      = np.asarray(pi_hat,      dtype=np.float32)
    pi_test_f32 = np.asarray(pi_hat_test, dtype=np.float32)
    return BCFModel(
        _fit_warmstart_bcf(X, y_scaled, z_f32, pi_f32, X_test, pi_test_f32,
                           n_gfr_burnin, n_mcmc_burnin, n_samples,
                           seed, keep_gfr_samples, config),
        y_mean, y_sd)


__all__ = [
    "BARTConfig", "BARTModel", "fit_bart", "fit_xbart", "fit_warmstart_bart",
    "BCFConfig", "BCFModel", "fit_bcf", "fit_xbcf", "fit_warmstart_bcf",
]
