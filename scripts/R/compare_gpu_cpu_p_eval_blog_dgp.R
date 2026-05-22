library(faststochtree)

# ── DGP from blog post ────────────────────────────────────────────────────────
# y = sin(2π x₁) + N(0, 1),  X ~ U[0, 1]^(n × p)
# n_train = 50 000,  p = 50,  T = 200 trees
#
# GPU conditions run first so the chip is cold and unthrottled.
# Note: back-to-back GPU runs at this scale heat the chip; the long CPU runs
# that follow give time for it to cool, so CPU numbers are also fair.
# ─────────────────────────────────────────────────────────────────────────────
set.seed(99)
n_train   <- 50000L
n_test    <- 500L
p         <- 50L
T         <- 200L
n_burnin  <- 5L
n_samples <- 10L     # 15 total sweeps; increase for tighter RMSE
seed      <- 42L

# Use up to 8 physical cores for the CPU path (blog post used 8).
# Reduce if your machine has fewer cores or you want a single-threaded baseline.
n_threads <- min(parallel::detectCores(logical = FALSE), 8L)

n     <- n_train + n_test
X     <- matrix(runif(n * p), n, p)
y     <- sin(2 * pi * X[, 1]) + rnorm(n)

X_train <- X[seq_len(n_train), ];           y_train <- y[seq_len(n_train)]
X_test  <- X[n_train + seq_len(n_test), ];  y_test  <- y[n_train + seq_len(n_test)]

# ── Conditions ────────────────────────────────────────────────────────────────
conditions <- list(
  list(label = "GPU  p_eval=0 ", gpu = TRUE,  p_eval = 0L,  threads = 1L),
  list(label = "GPU  p_eval=10", gpu = TRUE,  p_eval = 10L, threads = 1L),
  list(label = "CPU  p_eval=0 ", gpu = FALSE, p_eval = 0L,  threads = n_threads),
  list(label = "CPU  p_eval=10", gpu = FALSE, p_eval = 10L, threads = n_threads)
)

rmse <- function(pred, truth) sqrt(mean((pred - truth)^2))

# ── Runs ──────────────────────────────────────────────────────────────────────
cat(sprintf(
  "\nDGP: y = sin(2π x₁) + N(0,1),  X ~ U[0,1]^(n×p)\n"
))
cat(sprintf(
  "n_train=%d  n_test=%d  p=%d  T=%d  sweeps=%d  cpu_threads=%d\n\n",
  n_train, n_test, p, T, n_burnin + n_samples, n_threads
))
cat(sprintf("%-18s  %8s  %8s\n", "condition", "time (s)", "test RMSE"))
cat(strrep("-", 42), "\n")

results <- list()
for (cond in conditions) {
  cfg <- bart_config(num_trees  = T,
                     p_eval     = cond$p_eval,
                     num_threads = cond$threads)

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
  cat(sprintf("%-18s  %8.1f  %8.4f\n", cond$label, elapsed, err))
}

# ── Summary ratios ─────────────────────────────────────────────────────────────
cat("\n")
gpu0  <- results[["GPU  p_eval=0 "]]$time
gpu10 <- results[["GPU  p_eval=10"]]$time
cpu0  <- results[["CPU  p_eval=0 "]]$time
cpu10 <- results[["CPU  p_eval=10"]]$time

cat(sprintf("GPU  p_eval speedup  (0→10):               %.2fx\n", gpu0  / gpu10))
cat(sprintf("CPU  p_eval speedup  (0→10):               %.2fx\n", cpu0  / cpu10))
cat(sprintf("GPU vs CPU speedup   (p_eval=0,  %d thr): %.2fx\n", n_threads, cpu0  / gpu0))
cat(sprintf("GPU vs CPU speedup   (p_eval=10, %d thr): %.2fx\n", n_threads, cpu10 / gpu10))
cat("\n")
