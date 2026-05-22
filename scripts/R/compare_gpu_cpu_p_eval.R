library(faststochtree)

# ── Data ──────────────────────────────────────────────────────────────────────
set.seed(99)
n_train <- 5000
n_test  <- 1000
p       <- 30

n <- n_train + n_test
X <- matrix(rnorm(n * p), n, p)
# Signal in first 8 features; remaining 22 are noise.
y <- sin(X[, 1]) + 0.8 * X[, 2] - 0.5 * X[, 3] * X[, 4] +
     0.4 * X[, 5]^2 + 0.3 * rowSums(X[, 6:8]) + rnorm(n, sd = 0.5)

X_train <- X[seq_len(n_train), ];          y_train <- y[seq_len(n_train)]
X_test  <- X[n_train + seq_len(n_test), ]; y_test  <- y[n_train + seq_len(n_test)]

# ── Configuration ─────────────────────────────────────────────────────────────
n_burnin  <- 10L
n_samples <- 20L   # 30 total sweeps
seed      <- 42L
T         <- 50L

# GPU conditions run first: Metal dispatch overhead dominates at n=5000, so
# the p_eval GPU speedup is small (roundtrip cost, not kernel cost). At larger
# n the kernel-time savings dominate and GPU p_eval wins more clearly.
# Running GPU first also avoids thermal throttling from long CPU runs.
conditions <- list(
  list(label = "GPU  p_eval=0 ", gpu = TRUE,  p_eval = 0L),
  list(label = "GPU  p_eval=10", gpu = TRUE,  p_eval = 10L),
  list(label = "CPU  p_eval=0 ", gpu = FALSE, p_eval = 0L),
  list(label = "CPU  p_eval=10", gpu = FALSE, p_eval = 10L)
)

rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

# ── Runs ──────────────────────────────────────────────────────────────────────
cat(sprintf("\nn_train=%d  n_test=%d  p=%d  T=%d  n_burnin=%d  n_samples=%d\n",
            n_train, n_test, p, T, n_burnin, n_samples))
cat(sprintf("Signal in features 1-8 of %d  |  p_eval=10 evaluates %.0f%% of features per split\n\n",
            p, 100 * 10 / p))
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
                   gpu       = cond$gpu,
                   config    = cfg)
  elapsed <- proc.time()[["elapsed"]] - t0

  preds <- colMeans(test_samples(fit))
  err   <- rmse(preds, y_test)
  results[[cond$label]] <- list(time = elapsed, rmse = err)
  cat(sprintf("%-18s  %8.2f  %8.4f\n", cond$label, elapsed, err))
}

# ── Summary ratios ─────────────────────────────────────────────────────────────
cat("\n")
cpu0  <- results[["CPU  p_eval=0 "]]$time
cpu10 <- results[["CPU  p_eval=10"]]$time
gpu0  <- results[["GPU  p_eval=0 "]]$time
gpu10 <- results[["GPU  p_eval=10"]]$time

cat(sprintf("CPU  p_eval speedup  (0→10):       %.2fx\n", cpu0  / cpu10))
cat(sprintf("GPU  p_eval speedup  (0→10):       %.2fx\n", gpu0  / gpu10))
cat(sprintf("GPU vs CPU speedup   (p_eval=0):   %.2fx\n", cpu0  / gpu0))
cat(sprintf("GPU vs CPU speedup   (p_eval=10):  %.2fx\n", cpu10 / gpu10))
cat("\n")
