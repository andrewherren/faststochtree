library(faststochtree)

# ── CPU benchmark — blog DGP ───────────────────────────────────────────────────
# DGP: y = sin(2π x₁) + N(0,1),  X ~ U[0,1]^(n×p)
# n_train=50000  p=50  T=200
#
# Run this script in a SEPARATE fresh terminal session from bench_gpu_blog_dgp.R.
# Why separate: sustained GPU work heats the M-series chip enough to throttle
# CPU performance by 2–4×. Running after even ~30s of GPU activity gives
# misleading CPU timings. A fresh session with no prior heavy compute is the
# only reliable baseline.
#
# How we verify there is no thermal throttling:
#   We time two consecutive 2-sweep probe runs (1 thread, so thread-count
#   variance doesn't confuse the check). On a stable chip the second should be
#   within ~15% of the first. A ratio > 1.30 means the chip is warm; wait
#   ~2 minutes and re-run.
# ─────────────────────────────────────────────────────────────────────────────
set.seed(99)
n_train   <- 50000L
n_test    <- 500L
p         <- 50L
T         <- 200L
n_burnin  <- 5L
n_samples <- 10L
seed      <- 42L

# Use up to 8 physical cores (blog post used 8).
n_threads <- min(parallel::detectCores(logical = FALSE), 8L)

n     <- n_train + n_test
X     <- matrix(runif(n * p), n, p)
y     <- sin(2 * pi * X[, 1]) + rnorm(n)
X_train <- X[seq_len(n_train), ];           y_train <- y[seq_len(n_train)]
X_test  <- X[n_train + seq_len(n_test), ];  y_test  <- y[n_train + seq_len(n_test)]

rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

# Probe uses 1 thread: threading variance would widen the ratio band and
# mask genuine throttling.
probe_cpu_1thr <- function() {
  cfg <- bart_config(num_trees = T, p_eval = 0L, num_threads = 1L)
  t0  <- proc.time()[["elapsed"]]
  fit_xbart(X_train, y_train, X_test, n_burnin = 1L, n_samples = 1L,
            seed = seed, gpu = FALSE, config = cfg)
  proc.time()[["elapsed"]] - t0
}

# ── Thermal probe ─────────────────────────────────────────────────────────────
cat("Thermal probe (2 × 2-sweep CPU run, 1 thread)... ")
t1 <- probe_cpu_1thr()
t2 <- probe_cpu_1thr()
ratio <- t2 / t1
cat(sprintf("%.1fs → %.1fs (ratio %.2f)\n", t1, t2, ratio))
if (ratio > 1.30) {
  stop(sprintf(
    "Chip appears to be throttling (2nd run %.0f%% slower than 1st).\n%s",
    (ratio - 1) * 100,
    "Wait ~2 minutes for the chip to cool, then re-run."
  ))
}
cat("Thermal check passed — chip is in a stable state.\n\n")

# ── Conditions ────────────────────────────────────────────────────────────────
conditions <- list(
  list(label = "CPU  p_eval=0 ", p_eval = 0L),
  list(label = "CPU  p_eval=10", p_eval = 10L)
)

cat(sprintf(
  "DGP: y = sin(2π x₁) + N(0,1),  X ~ U[0,1]^(n×p)\n"
))
cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  T=%d  sweeps=%d  cpu_threads=%d\n\n",
  n_train, n_test, p, T, n_burnin + n_samples, n_threads
))
cat(sprintf("%-18s  %8s  %8s\n", "condition", "time (s)", "test RMSE"))
cat(strrep("-", 42), "\n")

results <- list()
for (cond in conditions) {
  cfg <- bart_config(num_trees = T, p_eval = cond$p_eval, num_threads = n_threads)
  t0  <- proc.time()[["elapsed"]]
  fit <- fit_xbart(X_train, y_train, X_test,
                   n_burnin  = n_burnin,
                   n_samples = n_samples,
                   seed      = seed,
                   gpu       = FALSE,
                   config    = cfg)
  elapsed <- proc.time()[["elapsed"]] - t0
  preds   <- colMeans(test_samples(fit))
  err     <- rmse(preds, y_test)
  results[[cond$label]] <- list(time = elapsed, rmse = err)
  cat(sprintf("%-18s  %8.1f  %8.4f\n", cond$label, elapsed, err))
}

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\n")
cpu0  <- results[["CPU  p_eval=0 "]]$time
cpu10 <- results[["CPU  p_eval=10"]]$time
cat(sprintf("CPU  p_eval speedup  (0→10):  %.2fx\n", cpu0 / cpu10))
cat(sprintf("(Run bench_gpu_blog_dgp.R in a separate session to compare GPU vs CPU)\n"))
cat("\n")
