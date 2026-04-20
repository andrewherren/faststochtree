#!/usr/bin/env Rscript
# bench/bench_r_packages_gfr.R — compare stochtree (GFR) and XBART R package
#
# DGP: y = sin(2π·x₁) + N(0, sigma_true²),  X ~ U[0,1]^(n×p)
#
# GFR schedule: 40 total sweeps, 15 burnin, 25 retained  (matches faststochtree bench)
#   stochtree: num_gfr=40, num_burnin=0, num_mcmc=0  (predictions from last GFR sweep)
#   XBART:     num_sweeps=40, burnin=15              (predictions averaged over sweeps 16-40)
#
# Each iteration uses independent seeds:
#   data seed  = seed + (iter - 1)
#   model seed = seed + n_iters + (iter - 1)
#
# Usage:
#   Rscript bench/bench_r_packages_gfr.R
#   Rscript bench/bench_r_packages_gfr.R --n_train 50000 --p 50 --iters 10

# ── Parse arguments ───────────────────────────────────────────────────────────
args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(args, key, default) {
  idx <- which(args == paste0("--", key))
  if (length(idx) == 1L && idx < length(args)) return(args[idx + 1L])
  default
}

n_train    <- as.integer(get_arg(args, "n_train",  "50000"))
n_test     <- as.integer(get_arg(args, "n_test",   "500"))
p          <- as.integer(get_arg(args, "p",        "50"))
num_trees  <- as.integer(get_arg(args, "trees",    "200"))
n_sweeps   <- as.integer(get_arg(args, "sweeps",   "40"))
n_burnin   <- as.integer(get_arg(args, "burnin",   "15"))
seed       <- as.integer(get_arg(args, "seed",     "12345"))
sigma_true <- as.numeric(get_arg(args, "sigma",    "1.0"))
n_iters    <- as.integer(get_arg(args, "iters",    "10"))
csv_path   <- get_arg(args, "csv", "bench/results/bench_results_r_gfr.csv")

n_retained <- n_sweeps - n_burnin

cat(sprintf("R package GFR benchmark: stochtree, XBART\n"))
cat(sprintf("  n_train=%d  n_test=%d  p=%d  trees=%d  iters=%d\n",
            n_train, n_test, p, num_trees, n_iters))
cat(sprintf("  sweeps=%d  burnin=%d  retained=%d  sigma_true=%.2f\n\n",
            n_sweeps, n_burnin, n_retained, sigma_true))

library(stochtree)
library(XBART)

# ── CSV setup ─────────────────────────────────────────────────────────────────
dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)
csv_is_new <- !file.exists(csv_path) || file.size(csv_path) == 0
csv_con <- file(csv_path, open = "a")
if (csv_is_new)
  writeLines("tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post",
             csv_con)

write_csv_row <- function(tag, iter, ms, rmse) {
  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  writeLines(sprintf("%s,%s,gfr,%d,%d,%d,%d,%d,1,%d,%d,%d,%.4f,NA",
                     tag, ts, n_train, p, num_trees,
                     n_burnin, n_retained, iter, n_iters, ms, rmse),
             csv_con)
  flush(csv_con)
}

st_ms   <- numeric(n_iters);  st_rmse   <- numeric(n_iters)
xb_ms   <- numeric(n_iters);  xb_rmse   <- numeric(n_iters)

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

  cat(sprintf("iter %2d/%d\n", iter, n_iters))

  # ── stochtree (GFR) ──────────────────────────────────────────────────────────
  cat(sprintf("  stochtree GFR ..."))
  flush(stdout())

  t0 <- proc.time()[["elapsed"]]
  st_result <- bart(
    X_train            = X_train,
    y_train            = y_train,
    X_test             = X_test,
    num_gfr            = n_sweeps,
    num_burnin         = 0,
    num_mcmc           = 0,
    general_params     = list(random_seed           = model_seed),
    mean_forest_params = list(num_trees              = num_trees,
                              num_features_subsample = round(sqrt(p)))
  )
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  # stochtree GFR returns predictions from the final sweep
  yhat_st  <- rowMeans(st_result$y_hat_test)
  rmse_st  <- sqrt(mean((y_test - yhat_st)^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_st))
  st_ms[iter] <- ms;  st_rmse[iter] <- rmse_st
  write_csv_row("stochtree-gfr", iter, ms, rmse_st)

  # ── XBART ────────────────────────────────────────────────────────────────────
  cat(sprintf("  XBART         ..."))
  flush(stdout())

  t0 <- proc.time()[["elapsed"]]
  xb_fit <- XBART::XBART(
    y            = matrix(y_train, ncol = 1L),
    X            = X_train,
    num_trees    = num_trees,
    num_sweeps   = n_sweeps,
    burnin       = n_burnin,
    random_seed  = model_seed,
    mtry         = round(sqrt(p)),
    verbose      = FALSE
  )
  yhat_xb <- predict(xb_fit, X_test)
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  # predict() returns a (n_retained x n_test) matrix; average over retained sweeps
  if (is.matrix(yhat_xb)) yhat_xb <- rowMeans(yhat_xb)
  rmse_xb <- sqrt(mean((y_test - yhat_xb)^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_xb))
  xb_ms[iter] <- ms;  xb_rmse[iter] <- rmse_xb
  write_csv_row("XBART", iter, ms, rmse_xb)

  cat("\n")
}

close(csv_con)

# ── Summary table ─────────────────────────────────────────────────────────────
cat(sprintf("Average over %d iterations:\n\n", n_iters))
cat(sprintf("%-20s  %9s  %8s\n", "package", "time (ms)", "RMSE"))
cat(sprintf("%-20s  %9s  %8s\n", "--------------------", "---------", "--------"))
cat(sprintf("%-20s  %9d  %8.4f\n", "stochtree (GFR)",
            as.integer(round(mean(st_ms))), mean(st_rmse)))
cat(sprintf("%-20s  %9d  %8.4f\n", "XBART",
            as.integer(round(mean(xb_ms))), mean(xb_rmse)))
cat(sprintf("\nCSV: %s\n", csv_path))
