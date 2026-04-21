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
