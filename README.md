# Fast Stochastic Forest Algorithms

Inspired by Giacomo Petrillo's JAX-based [bartz](https://github.com/bartz-org/bartz) GPU-accelerated BART project, `faststochtree` provides an optimized implementation of the BART MCMC and grow-from-root (GFR/XBART) algorithms, with a focus on Apple Silicon hardware.

## Why Apple Silicon?

M-series machines are (a) high-performance consumer devices, and (b) very common in tech and academia. A BART implementation that makes efficient use of this platform would allow for fast model development and inference without cloud-based Nvidia GPU servers.

## Why a new project?

[stochtree](https://github.com/StochasticTree/stochtree) was built to be a robust library for flexible and composable BART modeling. However, its expressiveness means a large surface of supported features, which affords fewer opportunities for performance optimization. Starting from the simplest possible BART model provides an ideal testbed for fine-tuning performance.

---

## Getting Started

Requires CMake ≥ 3.20 and a C++17 compiler. An internet connection is needed on first build to fetch GoogleTest.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(sysctl -n hw.logicalcpu)
```

---

## Benchmark binary

One of the standard build artifacts for `faststochtree` (crucial to its development) is a benchmarking binary that fits a BART / XBART model to generated dataset. After building, the program can be run as follows.

```bash
./build-release/faststochtree_bench [options]
```

**DGP:** `y = sin(2π·x₁) + N(0, σ²)`, `X ~ U[0,1]^(n×p)`

| Option | Default | Description |
|---|---|---|
| `--mode` | `bart` | `bart` (MCMC) or `xbart` (GFR) |
| `--n_train` | `50000` | Training set size |
| `--n_test` | `500` | Test set size |
| `--p` | `50` | Number of covariates |
| `--trees` | `200` | Trees in the ensemble |
| `--burnin` | `200` / `15` | Burn-in sweeps (bart / xbart defaults) |
| `--samples` | `1000` / `25` | Posterior samples (bart / xbart defaults) |
| `--threads` | `1` | Thread count (relevant for xbart) |
| `--iters` | `10` | Independent repetitions (different seeds) |
| `--seed` | `12345` | Base random seed |
| `--sigma` | `1.0` | True noise standard deviation |
| `--csv` | `bench/results/bench_results.csv` | Append per-iteration rows to this file |

Each of the `--iters` repetitions uses independent data and model seeds derived from `--seed`, so timing and RMSE can be averaged over independent data realisations.

To run a quick "smoke-test":

```bash
./build-release/faststochtree_bench --n_train 2000 --p 10 --trees 50 --burnin 50 --samples 100 --iters 3
```

---

## Benchmark suite — `bench/run_rmse_experiment.sh`

This script benchmarks one or more of the git tags that define the scaffolded rollout of performance improvements across a grid of `(n, p)` values. The `run_rmse_experiment` script runs the algorithm `K` times each against `M` independently-seeded synthetic datasets, writing one row per (dataset, chain) to `bench/results/bench_results.csv`. This separates dataset variance from sampler variance for a fair comparison across versions.

```bash
# Current checkout, default scenario (n=50k, p=50, M=10 datasets, K=5 chains)
./bench/run_rmse_experiment.sh

# Specific tags via git worktrees (your working tree is untouched)
./bench/run_rmse_experiment.sh --tags v12-leaf-counts,v13-flat-obs

# XBART mode
./bench/run_rmse_experiment.sh --mode xbart --tags gfr-v9-histogram,gfr-v10-flat-node-range

# Guard against slow early versions
./bench/run_rmse_experiment.sh --tags v1-naive-cpp --timeout 120

# Tiny run for smoke-testing
./bench/run_rmse_experiment.sh --quick
```

| Option | Default | Description |
|---|---|---|
| `--tags TAG[,...]` | current checkout | Git tags to benchmark via worktrees |
| `--M N` | `10` | Number of independent datasets |
| `--K N` | `5` | Chains per dataset |
| `--n N[,N,...]` | `50000` | Training-set sizes |
| `--p P[,P,...]` | `50` | Feature counts |
| `--mode` | `bart` | `bart` or `xbart` |
| `--trees` | `200` | Trees in ensemble |
| `--burnin` | `200`/`15` | Burn-in iterations (bart / xbart) |
| `--samples` | `1000`/`25` | Posterior samples (bart / xbart) |
| `--timeout SECS` | `0` (none) | Kill a run after N seconds |
| `--results FILE` | `bench/results/bench_results.csv` | Output CSV |
| `--no-skip` | — | Re-run even if result already recorded |
| `--quick` | — | Tiny grid for smoke-testing |

> **Note on slow tags:** v1 through v6 (BART) and gfr-v3 through gfr-v8 (GFR) can each take several minutes per run at n=50k. Use `--timeout` or restrict `--n` when benchmarking early versions.

---

## Results files

| File | Written by | Content |
|---|---|---|
| `bench/results/bench_results.csv` | `run_rmse_experiment.sh`, C++ binary | One row per (tag, dataset, chain) |
| `bench/results/bench_results_r.csv` | `bench_r_packages.R` | One row per iteration (R MCMC packages) |
| `bench/results/bench_results_r_gfr.csv` | `bench_r_packages_gfr.R` | One row per iteration (R GFR packages) |

---

## Plotting results

The plots in the blog post are created via a python script that reads from `bench_results.csv`

```bash
# Default (uses bench/results/bench_results.csv)
python bench/plot_results.py

# Explicit CSV path
python bench/plot_results.py --csv bench/results/bench_results.csv

# Print aggregated results to terminal as well
python bench/plot_results.py --show

# Restrict to a specific scenario
python bench/plot_results.py --n 200000 --p 10
```

This produces four PNGs in `bench/results/`:

| File | Content |
|---|---|
| `speedup_bart.png` | BART MCMC wall-time by version |
| `speedup_gfr.png` | GFR/XBART wall-time by version |
| `rmse_bart.png` | BART MCMC test RMSE by version |
| `rmse_gfr.png` | GFR/XBART test RMSE by version |

---

## Cross-implementation comparisons

All comparison scripts use the same DGP (`y = sin(2π·x₁) + N(0, σ²)`) and run `--iters` independent repetitions, reporting per-iteration timing and RMSE then an average. RNG implementations differ across languages, so individual data realisations will not match, but the DGP structure is identical.

Results from the R scripts are written to separate CSVs (separate from the C++ CSV). Pass `--csv FILE` to override.

### MCMC comparison — `bench/bench_r_packages.R`

Times `dbarts`, `BART::wbart`, `flexBART`, and `stochtree` back-to-back on the same data each iteration. Results written to `bench/results/bench_results_r.csv`.

```bash
Rscript bench/bench_r_packages.R
Rscript bench/bench_r_packages.R --n_train 10000 --p 10 --iters 5
```

### GFR comparison — `bench/bench_r_packages_gfr.R`

Times `stochtree` (GFR mode) and `XBART` back-to-back on the same data each iteration. Results written to `bench/results/bench_results_r_gfr.csv`.

```bash
Rscript bench/bench_r_packages_gfr.R
Rscript bench/bench_r_packages_gfr.R --n_train 10000 --p 10 --iters 5
```

### bartz (JAX/GPU) — `bench/bench_bartz.ipynb`

Jupyter notebook intended for Colab GPU. Set `n_iters` in the configuration cell. The first iteration includes JAX JIT compilation; the summary reports averages both with and without it.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewherren/faststochtree/blob/main/bench/bench_bartz.ipynb)

---

## Unit tests

`faststochtree` also includes a test suite which can be run either via `ctest`

```bash
ctest --test-dir build-release --output-on-failure
```

or by executing the test program directly

```bash
./build-release/test_bart
```
