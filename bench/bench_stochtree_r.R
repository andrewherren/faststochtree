#!/usr/bin/env Rscript
# bench/bench_stochtree_r.R — time stochtree R BART/XBART on the same DGP as bench/main.cpp
#
# DGP: y = sin(2π·x₁) + N(0, sigma_true²),  X ~ U[0,1]^(n×p)
#
# Hyperparameters matched to stochtree_compare.cpp:
#   T=200, alpha=0.95, beta=2.0, min_node_size=5, max_depth=6
#   tau = 0.25/T, sigma² ~ IG(1.5, 1.5·sigma_true²)
#
# Each iteration uses independent seeds:
#   data seed  = seed + (iter - 1)
#   model seed = seed + n_iters + (iter - 1)
#
# Usage:
#   Rscript bench/bench_stochtree_r.R
#   Rscript bench/bench_stochtree_r.R --mode xbart
#   Rscript bench/bench_stochtree_r.R --n_train 10000 --p 10 --iters 5

# ── Parse arguments ───────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(args, key, default) {
  idx <- which(args == paste0("--", key))
  if (length(idx) == 1L && idx < length(args)) return(args[idx + 1L])
  default
}

mode       <- get_arg(args, "mode",    "bart")
xbart      <- (mode == "xbart")

n_train    <- as.integer(get_arg(args, "n_train",  "50000"))
n_test     <- as.integer(get_arg(args, "n_test",   "500"))
p          <- as.integer(get_arg(args, "p",         "50"))
num_trees  <- as.integer(get_arg(args, "trees",     "200"))
n_gfr      <- as.integer(get_arg(args, "gfr",      if (xbart) "40"  else "0"))
n_burnin   <- as.integer(get_arg(args, "burnin",    if (xbart) "0"   else "200"))
n_samples  <- as.integer(get_arg(args, "samples",   if (xbart) "0"   else "1000"))
seed       <- as.integer(get_arg(args, "seed",      "12345"))
sigma_true <- as.numeric(get_arg(args, "sigma",     "1.0"))
n_iters    <- as.integer(get_arg(args, "iters",     "10"))
csv_path   <- get_arg(args, "csv", "bench/results/bench_results_r.csv")

num_features_subsample <- if (xbart) as.integer(floor(sqrt(p))) else p

tau      <- 0.25 / num_trees
a_global <- 1.5
b_global <- 1.5 * sigma_true^2

label <- if (xbart) "XBART (GFR)" else "BART (MCMC)"
cat(sprintf("stochtree R %s benchmark\n", label))
cat(sprintf("  n_train=%d  n_test=%d  p=%d  trees=%d  iters=%d\n",
            n_train, n_test, p, num_trees, n_iters))
if (xbart) {
  cat(sprintf("  gfr=%d  feature_subsample=%d  sigma_true=%.2f\n\n",
              n_gfr, num_features_subsample, sigma_true))
} else {
  cat(sprintf("  burnin=%d  samples=%d  sigma_true=%.2f\n\n",
              n_burnin, n_samples, sigma_true))
}

library(stochtree)

# ── CSV setup ─────────────────────────────────────────────────────────────────
csv_tag <- if (xbart) "stochtree-r-xbart" else "stochtree-r-bart"
dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)
csv_is_new <- !file.exists(csv_path) || file.size(csv_path) == 0
csv_con <- file(csv_path, open = "a")
if (csv_is_new)
  writeLines("tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post",
             csv_con)

ms_vec    <- numeric(n_iters)
rmse_vec  <- numeric(n_iters)

for (iter in seq_len(n_iters)) {
  data_seed  <- seed + (iter - 1L)
  model_seed <- seed + n_iters + (iter - 1L)

  # ── Simulate data ────────────────────────────────────────────────────────────
  set.seed(data_seed)
  n_total <- n_train + n_test
  X_all   <- matrix(runif(n_total * p), nrow = n_total, ncol = p)
  y_all   <- sin(2 * pi * X_all[, 1]) + rnorm(n_total, sd = sigma_true)

  X_train <- X_all[seq_len(n_train), , drop = FALSE]
  y_train <- y_all[seq_len(n_train)]
  X_test  <- X_all[(n_train + 1L):n_total, , drop = FALSE]
  y_test  <- y_all[(n_train + 1L):n_total]

  cat(sprintf("iter %2d/%d: sampling...", iter, n_iters))
  flush(stdout())

  t0 <- proc.time()[["elapsed"]]

  bart_result <- bart(
    X_train            = X_train,
    y_train            = y_train,
    X_test             = X_test,
    num_gfr            = n_gfr,
    num_burnin         = n_burnin,
    num_mcmc           = n_samples,
    general_params     = list(random_seed           = model_seed,
                              sigma2_global_shape    = a_global,
                              sigma2_global_scale    = b_global),
    mean_forest_params = list(num_trees              = num_trees,
                              alpha                  = 0.95,
                              beta                   = 2.0,
                              min_node_size          = 5L,
                              max_depth              = 6L,
                              sigma2_leaf_init       = tau,
                              num_features_subsample = num_features_subsample)
  )

  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  y_hat <- rowMeans(bart_result$y_hat_test)
  rmse  <- sqrt(mean((y_test - y_hat)^2))

  cat(sprintf("  %6d ms  RMSE=%.4f\n", ms, rmse))
  ms_vec[iter]   <- ms
  rmse_vec[iter] <- rmse

  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  writeLines(sprintf("%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,NA",
                     csv_tag, ts, mode, n_train, p, num_trees,
                     n_burnin, n_samples, 1L, iter, n_iters, ms, rmse),
             csv_con)
  flush(csv_con)
}

close(csv_con)

cat(sprintf("\nAverage over %d iterations:\n", n_iters))
cat(sprintf("  time: %d ms\n",  as.integer(round(mean(ms_vec)))))
cat(sprintf("  RMSE: %.4f\n",   mean(rmse_vec)))
cat(sprintf("  CSV:  %s\n",     csv_path))
