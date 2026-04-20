library(dplyr)

r_pkgs <- read.csv("bench/results/bench_results_r.csv") |>
  select(tag, time_ms, rmse)

v13 <- read.csv("bench/results/bench_results_rmse.csv") |>
  filter(tag == "v13-flat-obs", n_train == 50000, p == 50) |>
  select(tag, time_ms, rmse)

dat <- bind_rows(r_pkgs, v13) |>
  mutate(
    tag = recode(tag,
      "v13-flat-obs" = "faststochtree (v13)",
      "BART-wbart"   = "BART",
      "dbarts"       = "dbarts",
      "flexBART"     = "flexBART"
    )
  )

summary <- dat |>
  group_by(tag) |>
  summarise(
    mean_s    = mean(time_ms) / 1000,
    median_s  = median(time_ms) / 1000,
    mean_rmse = mean(rmse),
    .groups   = "drop"
  ) |>
  arrange(mean_s)

cat("=== Runtime and RMSE summary (n=50k, p=50, 200 trees, 200 burnin + 1000 samples) ===\n\n")
for (i in seq_len(nrow(summary))) {
  r <- summary[i, ]
  cat(sprintf("%-25s  mean: %5.1fs  median: %5.1fs  mean RMSE: %.4f\n",
              r$tag, r$mean_s, r$median_s, r$mean_rmse))
}

cat("\n=== Speedup relative to each R package (vs faststochtree mean) ===\n\n")
ft_mean <- summary$mean_s[summary$tag == "faststochtree (v13)"]
for (i in seq_len(nrow(summary))) {
  r <- summary[i, ]
  if (r$tag != "faststochtree (v13)") {
    cat(sprintf("%-25s  %.1fx faster\n", r$tag, r$mean_s / ft_mean))
  }
}
