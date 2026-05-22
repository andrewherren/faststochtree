library(faststochtree)

# ── GPU benchmark — blog DGP ───────────────────────────────────────────────────
# DGP: y = sin(2π x₁) + N(0,1),  X ~ U[0,1]^(n×p)
# n_train=50000  p=50  T=200
#
# Run this script in a fresh terminal session before any sustained compute work.
# M-series chips throttle under load; a cold chip is essential for honest GPU
# numbers. See bench_cpu_blog_dgp.R for the CPU counterpart.
#
# How we verify there is no thermal throttling:
#   We time two consecutive 2-sweep probe runs. On a stable chip the second
#   should be within ~15% of the first. A ratio > 1.30 means the chip is
#   already warm; wait ~2 minutes and re-run.
# ─────────────────────────────────────────────────────────────────────────────
set.seed(99)
n_train   <- 50000L
n_test    <- 500L
p         <- 50L
T         <- 200L
n_burnin  <- 5L
n_samples <- 10L
seed      <- 42L

n     <- n_train + n_test
X     <- matrix(runif(n * p), n, p)
y     <- sin(2 * pi * X[, 1]) + rnorm(n)
X_train <- X[seq_len(n_train), ];           y_train <- y[seq_len(n_train)]
X_test  <- X[n_train + seq_len(n_test), ];  y_test  <- y[n_train + seq_len(n_test)]

rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

probe_gpu <- function() {
  cfg <- bart_config(num_trees = T, p_eval = 0L)
  t0  <- proc.time()[["elapsed"]]
  fit_xbart(X_train, y_train, X_test, n_burnin = 1L, n_samples = 1L,
            seed = seed, gpu = TRUE, config = cfg)
  proc.time()[["elapsed"]] - t0
}

# ── Thermal probe ─────────────────────────────────────────────────────────────
cat("Thermal probe (2 × 2-sweep GPU run)... ")
t1 <- probe_gpu()
t2 <- probe_gpu()
ratio <- t2 / t1
cat(sprintf("%.1fs → %.1fs (ratio %.2f)\n", t1, t2, ratio))
if (ratio > 1.30) {
  stop(sprintf(
    "Chip appears to be throttling (2nd run %.0f%% slower than 1st).\n",
    (ratio - 1) * 100,
    "Wait ~2 minutes for the chip to cool, then re-run."
  ))
}
cat("Thermal check passed — chip is in a stable state.\n\n")

# ── Conditions ────────────────────────────────────────────────────────────────
conditions <- list(
  list(label = "GPU  p_eval=0 ", p_eval = 0L),
  list(label = "GPU  p_eval=10", p_eval = 10L)
)

cat(sprintf(
  "DGP: y = sin(2π x₁) + N(0,1),  X ~ U[0,1]^(n×p)\n"
))
cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  T=%d  sweeps=%d\n\n",
  n_train, n_test, p, T, n_burnin + n_samples
))
cat(sprintf("%-18s  %8s  %8s\n", "condition", "time (s)", "test RMSE"))
cat(strrep("-", 42), "\n")

results <- list()
for (cond in conditions) {
  cfg <- bart_config(num_trees = T, p_eval = cond$p_eval)
  t0  <- proc.time()[["elapsed"]]
  fit <- fit_xbart(X_train, y_train, X_test,
                   n_burnin  = n_burnin,
                   n_samples = n_samples,
                   seed      = seed,
                   gpu       = TRUE,
                   config    = cfg)
  elapsed <- proc.time()[["elapsed"]] - t0
  preds   <- colMeans(test_samples(fit))
  err     <- rmse(preds, y_test)
  results[[cond$label]] <- list(time = elapsed, rmse = err)
  cat(sprintf("%-18s  %8.1f  %8.4f\n", cond$label, elapsed, err))
}

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\n")
gpu0  <- results[["GPU  p_eval=0 "]]$time
gpu10 <- results[["GPU  p_eval=10"]]$time
cat(sprintf("GPU  p_eval speedup  (0→10):  %.2fx\n", gpu0 / gpu10))
cat("\n")
