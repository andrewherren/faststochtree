#' Create a BARTConfig list
#'
#' @param num_trees        Number of trees (default 200)
#' @param alpha            Tree prior base (default 0.95)
#' @param beta             Tree prior power (default 2.0)
#' @param leaf_prior_var   Leaf value prior variance tau; calibrated based on num_trees when set to -1.0
#' @param sigma2_shape     Noise variance prior shape nu (default 3.0)
#' @param sigma2_scale     Noise variance prior scale lambda (default 1.0)
#' @param min_samples_leaf Minimum observations per leaf (default 5)
#' @param p_eval           Features evaluated per node; 0 = all (default 0)
#' @param num_threads      Thread count for GFR parallelism (default 1, ignored for BART, only used by XBART)
#' @return A named list suitable for passing to fit_bart() or fit_xbart()
#' @export
bart_config <- function(num_trees        = 200L,
                        alpha            = 0.95,
                        beta             = 2.0,
                        leaf_prior_var   = -1.0,
                        sigma2_shape     = 3.0,
                        sigma2_scale     = 1.0,
                        min_samples_leaf = 5L,
                        p_eval           = 0L,
                        num_threads      = 1L) {
  list(num_trees        = as.integer(num_trees),
       alpha            = as.double(alpha),
       beta             = as.double(beta),
       leaf_prior_var   = as.double(leaf_prior_var),
       sigma2_shape     = as.double(sigma2_shape),
       sigma2_scale     = as.double(sigma2_scale),
       min_samples_leaf = as.integer(min_samples_leaf),
       p_eval           = as.integer(p_eval),
       num_threads      = as.integer(num_threads))
}

.scale_y <- function(y) {
  y_mean <- mean(y)
  y_sd   <- sd(y)
  if (y_sd == 0) y_sd <- 1.0
  list(y_scaled = (y - y_mean) / y_sd,
       y_mean   = y_mean,
       y_sd     = y_sd)
}

#' Fit a BART model via MCMC
#'
#' @param X        Numeric matrix [n × p] of training covariates
#' @param y        Numeric vector [n] of training responses
#' @param X_test   Numeric matrix [n_test × p] for in-sample test predictions
#' @param n_burnin Number of burn-in sweeps (default 200)
#' @param n_samples Number of posterior samples to retain (default 1000)
#' @param seed     Integer random seed (default 42)
#' @param config   Named list from bart_config() (default bart_config())
#' @return A BARTModel object
#' @export
fit_bart <- function(X, y, X_test,
                     n_burnin  = 200L,
                     n_samples = 1000L,
                     seed      = 42L,
                     config    = bart_config()) {
  sc <- .scale_y(y)
  if (config$leaf_prior_var < 0) {
    config$leaf_prior_var <- 1.0 / config$num_trees
  }
  ptr <- fit_bart_cpp(as.matrix(X), sc$y_scaled, as.matrix(X_test),
                      as.integer(n_burnin), as.integer(n_samples),
                      as.integer(seed), config)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BARTModel")
}

#' Fit an XBART model via grow-from-root (GFR)
#'
#' @param X        Numeric matrix [n × p] of training covariates
#' @param y        Numeric vector [n] of training responses
#' @param X_test   Numeric matrix [n_test × p] for in-sample test predictions
#' @param n_burnin    Number of GFR burn-in sweeps (default 15)
#' @param n_samples   Number of GFR samples to retain (default 25)
#' @param seed        Integer random seed (default 42)
#' @param config      Named list from bart_config() (optional; overrides num_threads)
#' @return A BARTModel object
#' @export
fit_xbart <- function(X, y, X_test,
                      n_burnin    = 15L,
                      n_samples   = 25L,
                      seed        = 42L,
                      config      = bart_config()) {
  sc <- .scale_y(y)
  if (config$leaf_prior_var < 0) {
    config$leaf_prior_var <- 1.0 / config$num_trees
  }
  ptr <- fit_xbart_cpp(as.matrix(X), sc$y_scaled, as.matrix(X_test),
                       as.integer(n_burnin), as.integer(n_samples),
                       as.integer(seed), config)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BARTModel")
}

#' Fit a warm-start BART model (GFR burn-in followed by MCMC sampling)
#'
#' @param X              Numeric matrix [n × p] of training covariates
#' @param y              Numeric vector [n] of training responses
#' @param X_test         Numeric matrix [n_test × p] for in-sample test predictions
#' @param n_gfr_burnin   Number of GFR warm-up sweeps (default 20)
#' @param n_mcmc_burnin  Additional MCMC burn-in after GFR (default 0)
#' @param n_samples      Number of MCMC posterior samples to retain (default 200)
#' @param seed           Integer random seed (default 42)
#' @param keep_gfr_samples If TRUE, prepend GFR burn-in draws to the result (default FALSE)
#' @param config         Named list from bart_config()
#' @return A BARTModel object
#' @export
fit_warmstart_bart <- function(X, y, X_test,
                                n_gfr_burnin     = 20L,
                                n_mcmc_burnin    = 0L,
                                n_samples        = 200L,
                                seed             = 42L,
                                keep_gfr_samples = FALSE,
                                config           = bart_config()) {
  sc <- .scale_y(y)
  if (config$leaf_prior_var < 0) {
    config$leaf_prior_var <- 1.0 / config$num_trees
  }
  ptr <- fit_warmstart_bart_cpp(as.matrix(X), sc$y_scaled, as.matrix(X_test),
                                 as.integer(n_gfr_burnin), as.integer(n_mcmc_burnin),
                                 as.integer(n_samples), as.integer(seed),
                                 as.logical(keep_gfr_samples), config)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BARTModel")
}

#' Posterior predictive samples for new observations
#'
#' @param object  A BARTModel object returned by fit_bart() or fit_xbart()
#' @param newdata Numeric matrix [n_new × p] of new covariates
#' @param ...     Ignored
#' @return Numeric matrix [n_samples × n_new] of posterior predictive draws
#' @export
predict.BARTModel <- function(object, newdata, ...) {
  predict_cpp(object$ptr, as.matrix(newdata)) * object$y_sd + object$y_mean
}

#' Posterior test predictions from the fit call
#'
#' @param model A BARTModel object
#' @return Numeric matrix [n_samples × n_test]
#' @export
test_samples <- function(model) {
  test_samples_cpp(model$ptr) * model$y_sd + model$y_mean
}

#' Posterior noise-variance samples
#'
#' @param model A BARTModel object
#' @return Numeric vector [n_samples]
#' @export
sigma2_samples <- function(model) {
  sigma2_samples_cpp(model$ptr) * model$y_sd^2
}

# ── BCF ───────────────────────────────────────────────────────────────────────

#' Create a BCFConfig list
#'
#' @param mu_config  Named list from bart_config() for the prognostic forest
#' @param tau_config Named list from bart_config() for the treatment-effect forest
#' @return A named list suitable for passing to fit_bcf() and related functions
#' @export
bcf_config <- function(mu_config  = bart_config(),
                       tau_config = bart_config(num_trees = 50L)) {
  list(mu = mu_config, tau = tau_config)
}

.scale_bcf_config <- function(config) {
  if (config$mu$leaf_prior_var  < 0) config$mu$leaf_prior_var  <- 1.0 / config$mu$num_trees
  if (config$tau$leaf_prior_var < 0) config$tau$leaf_prior_var <- 1.0 / config$tau$num_trees
  config
}

#' Fit a BCF model via MCMC
#'
#' @param X           Numeric matrix [n x p] of training covariates
#' @param y           Numeric vector [n] of training responses
#' @param z           Numeric vector [n] of binary treatment indicators (0/1)
#' @param pi_hat      Numeric vector [n] of estimated propensity scores
#' @param X_test      Numeric matrix [n_test x p] of test covariates
#' @param pi_hat_test Numeric vector [n_test] of test propensity scores
#' @param n_burnin    Number of MCMC burn-in sweeps (default 200)
#' @param n_samples   Number of posterior samples (default 1000)
#' @param seed        Integer random seed (default 42)
#' @param config      Named list from bcf_config()
#' @return A BCFModel object
#' @export
fit_bcf <- function(X, y, z, pi_hat, X_test, pi_hat_test = NULL,
                    n_burnin  = 200L, n_samples = 1000L,
                    seed = 42L, config = bcf_config()) {
  sc     <- .scale_y(y)
  config <- .scale_bcf_config(config)
  if (is.null(pi_hat_test)) pi_hat_test <- rep(0.0, nrow(X_test))
  ptr <- fit_bcf_cpp(as.matrix(X), sc$y_scaled, as.numeric(z), as.numeric(pi_hat),
                     as.matrix(X_test), as.numeric(pi_hat_test),
                     as.integer(n_burnin), as.integer(n_samples), as.integer(seed),
                     config$mu, config$tau)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BCFModel")
}

#' Fit a BCF model via GFR (XBCF)
#'
#' @param X           Numeric matrix [n x p] of training covariates
#' @param y           Numeric vector [n] of training responses
#' @param z           Numeric vector [n] of binary treatment indicators
#' @param pi_hat      Numeric vector [n] of estimated propensity scores
#' @param X_test      Numeric matrix [n_test x p] of test covariates
#' @param pi_hat_test Numeric vector [n_test] of test propensity scores
#' @param n_burnin    Number of GFR burn-in sweeps (default 15)
#' @param n_samples   Number of GFR samples (default 25)
#' @param seed        Integer random seed (default 42)
#' @param config      Named list from bcf_config()
#' @return A BCFModel object
#' @export
fit_xbcf <- function(X, y, z, pi_hat, X_test, pi_hat_test = NULL,
                     n_burnin  = 15L, n_samples = 25L,
                     seed = 42L, config = bcf_config()) {
  sc     <- .scale_y(y)
  config <- .scale_bcf_config(config)
  if (is.null(pi_hat_test)) pi_hat_test <- rep(0.0, nrow(X_test))
  ptr <- fit_xbcf_cpp(as.matrix(X), sc$y_scaled, as.numeric(z), as.numeric(pi_hat),
                      as.matrix(X_test), as.numeric(pi_hat_test),
                      as.integer(n_burnin), as.integer(n_samples), as.integer(seed),
                      config$mu, config$tau)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BCFModel")
}

#' Fit a warm-start BCF model (GFR burn-in followed by MCMC sampling)
#'
#' @param X              Numeric matrix [n x p] of training covariates
#' @param y              Numeric vector [n] of training responses
#' @param z              Numeric vector [n] of binary treatment indicators
#' @param pi_hat         Numeric vector [n] of estimated propensity scores
#' @param X_test         Numeric matrix [n_test x p] of test covariates
#' @param pi_hat_test    Numeric vector [n_test] of test propensity scores
#' @param n_gfr_burnin   Number of GFR warm-up sweeps (default 20)
#' @param n_mcmc_burnin  Additional MCMC burn-in after GFR (default 0)
#' @param n_samples      Number of MCMC posterior samples (default 200)
#' @param seed           Integer random seed (default 42)
#' @param keep_gfr_samples If TRUE, prepend GFR draws to result (default FALSE)
#' @param config         Named list from bcf_config()
#' @return A BCFModel object
#' @export
fit_warmstart_bcf <- function(X, y, z, pi_hat, X_test, pi_hat_test = NULL,
                               n_gfr_burnin  = 20L, n_mcmc_burnin = 0L,
                               n_samples = 200L, seed = 42L,
                               keep_gfr_samples = FALSE,
                               config = bcf_config()) {
  sc     <- .scale_y(y)
  config <- .scale_bcf_config(config)
  if (is.null(pi_hat_test)) pi_hat_test <- rep(0.0, nrow(X_test))
  ptr <- fit_warmstart_bcf_cpp(as.matrix(X), sc$y_scaled, as.numeric(z), as.numeric(pi_hat),
                                as.matrix(X_test), as.numeric(pi_hat_test),
                                as.integer(n_gfr_burnin), as.integer(n_mcmc_burnin),
                                as.integer(n_samples), as.integer(seed),
                                as.logical(keep_gfr_samples),
                                config$mu, config$tau)
  structure(list(ptr = ptr, y_mean = sc$y_mean, y_sd = sc$y_sd), class = "BCFModel")
}

#' Posterior CATE samples at test points
#'
#' @param model A BCFModel object
#' @return Numeric matrix [n_samples x n_test] of tau(x) draws
#' @export
test_tau_samples <- function(model) {
  test_tau_samples_cpp(model$ptr) * model$y_sd
}

#' Posterior prognostic effect samples at test points
#'
#' @param model A BCFModel object
#' @return Numeric matrix [n_samples x n_test] of mu(x) draws
#' @export
test_mu_samples <- function(model) {
  test_mu_samples_cpp(model$ptr) * model$y_sd + model$y_mean
}

#' Posterior noise-variance samples from a BCF model
#'
#' @param model A BCFModel object
#' @return Numeric vector [n_samples]
#' @export
sigma2_bcf_samples <- function(model) {
  sigma2_bcf_samples_cpp(model$ptr) * model$y_sd^2
}

#' Predict CATE tau(x) on new data
#'
#' @param object  A BCFModel object
#' @param newdata Numeric matrix [n_new x p] of new covariates
#' @param ...     Ignored
#' @return Numeric matrix [n_samples x n_new]
#' @export
predict_tau <- function(object, newdata, ...) {
  predict_tau_cpp(object$ptr, as.matrix(newdata)) * object$y_sd
}

#' Predict prognostic effect mu(x, pi_hat) on new data
#'
#' @param object   A BCFModel object
#' @param newdata  Numeric matrix [n_new x p] of new covariates
#' @param pi_hat   Numeric vector [n_new] of propensity scores for new data
#' @param ...      Ignored
#' @return Numeric matrix [n_samples x n_new]
#' @export
predict_mu <- function(object, newdata, pi_hat, ...) {
  X_mu <- cbind(as.matrix(newdata), as.numeric(pi_hat))
  predict_mu_cpp(object$ptr, X_mu) * object$y_sd + object$y_mean
}
