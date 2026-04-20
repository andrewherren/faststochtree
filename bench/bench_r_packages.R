#!/usr/bin/env Rscript
# bench/bench_r_packages.R — compare dbarts, BART::wbart, flexBART, and stochtree
#                            on the same DGP as bench/main.cpp (MCMC track only)
#
# DGP: y = sin(2π·x₁) + N(0, sigma_true²),  X ~ U[0,1]^(n×p)
#
# Hyperparameters — matched where both packages support it:
#   ntree=200, base/alpha=0.95, power/beta=2.0, ndpost=1000, nskip=200
#   k=2, sigdf=3, sigest=sigma_true
#
# Each iteration uses independent seeds:
#   data seed  = seed + (iter - 1)
#   model seed = seed + n_iters + (iter - 1)
#   (BART::wbart has no seed arg; seeded via set.seed() before each call)
#
# Usage:
#   Rscript bench/bench_r_packages.R
#   Rscript bench/bench_r_packages.R --n_train 10000 --p 10 --iters 5

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
n_burnin   <- as.integer(get_arg(args, "burnin",   "200"))
n_samples  <- as.integer(get_arg(args, "samples",  "1000"))
seed       <- as.integer(get_arg(args, "seed",     "12345"))
sigma_true <- as.numeric(get_arg(args, "sigma",    "1.0"))
n_iters    <- as.integer(get_arg(args, "iters",    "10"))
csv_path   <- get_arg(args, "csv", "bench/results/bench_results_r.csv")

cat(sprintf("R package BART benchmark (MCMC only): dbarts, BART::wbart, flexBART, stochtree\n"))
cat(sprintf("  n_train=%d  n_test=%d  p=%d  trees=%d  iters=%d\n",
            n_train, n_test, p, num_trees, n_iters))
cat(sprintf("  burnin=%d  samples=%d  sigma_true=%.2f\n\n",
            n_burnin, n_samples, sigma_true))

library(dbarts)
library(BART)
library(flexBART)
library(stochtree)

# ── CSV setup ─────────────────────────────────────────────────────────────────
dir.create(dirname(csv_path), recursive = TRUE, showWarnings = FALSE)
csv_is_new <- !file.exists(csv_path) || file.size(csv_path) == 0
csv_con <- file(csv_path, open = "a")
if (csv_is_new)
  writeLines("tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post",
             csv_con)

write_csv_row <- function(tag, iter, ms, rmse) {
  ts <- format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  writeLines(sprintf("%s,%s,bart,%d,%d,%d,%d,%d,1,%d,%d,%d,%.4f,NA",
                     tag, ts, n_train, p, num_trees,
                     n_burnin, n_samples, iter, n_iters, ms, rmse),
             csv_con)
  flush(csv_con)
}

db_ms   <- numeric(n_iters);  db_rmse   <- numeric(n_iters)
wb_ms   <- numeric(n_iters);  wb_rmse   <- numeric(n_iters)
fb_ms   <- numeric(n_iters);  fb_rmse   <- numeric(n_iters)
st_ms   <- numeric(n_iters);  st_rmse   <- numeric(n_iters)

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

  # ── dbarts::bart ─────────────────────────────────────────────────────────────
  cat(sprintf("  dbarts::bart ..."))
  flush(stdout())

  t0 <- proc.time()[["elapsed"]]
  db_result <- dbarts::bart(
    x.train   = X_train,
    y.train   = y_train,
    x.test    = X_test,
    ntree     = num_trees,
    ndpost    = n_samples,
    nskip     = n_burnin,
    k         = 2,
    power     = 2.0,
    base      = 0.95,
    sigdf     = 3.0,
    sigest    = sigma_true,
    nchain    = 1L,
    seed      = model_seed,
    verbose   = FALSE,
    keeptrees = TRUE
  )
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  yhat_db <- drop(db_result$yhat.test)   # (ndpost × n_test)
  rmse_db <- sqrt(mean((y_test - colMeans(yhat_db))^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_db))
  db_ms[iter] <- ms;  db_rmse[iter] <- rmse_db
  write_csv_row("dbarts", iter, ms, rmse_db)

  # ── BART::wbart ──────────────────────────────────────────────────────────────
  cat(sprintf("  BART::wbart   ..."))
  flush(stdout())

  set.seed(model_seed)   # wbart has no seed arg
  t0 <- proc.time()[["elapsed"]]
  wb_result <- BART::wbart(
    x.train    = X_train,
    y.train    = y_train,
    x.test     = X_test,
    ntree      = num_trees,
    ndpost     = n_samples,
    nskip      = n_burnin,
    k          = 2,
    power      = 2.0,
    base       = 0.95,
    sigdf      = 3L,
    sigest     = sigma_true,
    numcut     = 100L,
    printevery = .Machine$integer.max
  )
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  rmse_wb <- sqrt(mean((y_test - wb_result$yhat.test.mean)^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_wb))
  wb_ms[iter] <- ms;  wb_rmse[iter] <- rmse_wb
  write_csv_row("BART-wbart", iter, ms, rmse_wb)

  # ── flexBART ──────────────────────────────────────────────────────────────────
  cat(sprintf("  flexBART      ..."))
  flush(stdout())

  # flexBART uses a formula + data.frame interface
  col_names <- paste0("x", seq_len(p))
  train_df  <- as.data.frame(X_train); colnames(train_df) <- col_names
  test_df   <- as.data.frame(X_test);  colnames(test_df)  <- col_names
  train_df$y <- y_train

  fb_formula <- as.formula(paste("y ~ bart(", paste(col_names, collapse = " + "), ")"))

  set.seed(model_seed)
  t0 <- proc.time()[["elapsed"]]
  fb_result <- flexBART(
    formula          = fb_formula,
    train_data       = train_df,
    test_data        = test_df,
    M_vec            = num_trees,
    alpha_vec        = 0.95,
    beta_vec         = 2.0,
    nu               = 3,
    sigquant         = 0.9,
    sigest           = sigma_true,
    nd               = n_samples,
    burn             = n_burnin,
    n.chains         = 1L,
    sparse           = FALSE,
    verbose          = FALSE
  )
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  rmse_fb <- sqrt(mean((y_test - fb_result$yhat.test.mean)^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_fb))
  fb_ms[iter] <- ms;  fb_rmse[iter] <- rmse_fb
  write_csv_row("flexBART", iter, ms, rmse_fb)

  # ── stochtree::bart ───────────────────────────────────────────────────────────
  cat(sprintf("  stochtree     ..."))
  flush(stdout())

  t0 <- proc.time()[["elapsed"]]
  st_result <- stochtree::bart(
    X_train            = X_train,
    y_train            = y_train,
    X_test             = X_test,
    num_gfr            = 0L,
    num_burnin         = n_burnin,
    num_mcmc           = n_samples,
    general_params     = list(random_seed = model_seed),
    mean_forest_params = list(num_trees   = num_trees)
  )
  ms <- as.integer(round((proc.time()[["elapsed"]] - t0) * 1000))

  yhat_st <- rowMeans(st_result$y_hat_test)
  rmse_st <- sqrt(mean((y_test - yhat_st)^2))
  cat(sprintf(" %6d ms  RMSE=%.4f\n", ms, rmse_st))
  st_ms[iter] <- ms;  st_rmse[iter] <- rmse_st
  write_csv_row("stochtree-r-bart", iter, ms, rmse_st)

  cat("\n")
}

close(csv_con)

# ── Summary table ─────────────────────────────────────────────────────────────
cat(sprintf("Average over %d iterations:\n\n", n_iters))
cat(sprintf("%-20s  %9s  %8s\n", "package", "time (ms)", "RMSE"))
cat(sprintf("%-20s  %9s  %8s\n", "--------------------", "---------", "--------"))
cat(sprintf("%-20s  %9d  %8.4f\n", "dbarts::bart",
            as.integer(round(mean(db_ms))), mean(db_rmse)))
cat(sprintf("%-20s  %9d  %8.4f\n", "BART::wbart",
            as.integer(round(mean(wb_ms))), mean(wb_rmse)))
cat(sprintf("%-20s  %9d  %8.4f\n", "flexBART",
            as.integer(round(mean(fb_ms))), mean(fb_rmse)))
cat(sprintf("%-20s  %9d  %8.4f\n", "stochtree",
            as.integer(round(mean(st_ms))), mean(st_rmse)))
cat(sprintf("\nCSV: %s\n", csv_path))
