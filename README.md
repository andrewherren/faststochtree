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
| `--csv` | `bench/results/bench_results_mc.csv` | Append per-iteration rows to this file |

Each of the `--iters` repetitions uses independent data and model seeds derived from `--seed`, so timing and RMSE can be averaged over independent data realisations.

Quick smoke-test:

```bash
./build-release/faststochtree_bench --n_train 2000 --p 10 --trees 50 --burnin 50 --samples 100 --iters 3
```

---

## Benchmark suites

### Single-run sweep — `bench/run_suite.sh`

Benchmarks one or more git tags across a grid of `(n, p)` values, writing one row per run to `bench/results/bench_results.csv`. Useful for comparing algorithmic versions where a single timing is sufficient.

```bash
# Current checkout, default grid (n=10k,50k × p=10,50)
./bench/run_suite.sh

# Specific tags via git worktrees (your working tree is untouched)
./bench/run_suite.sh --tags v12-leaf-counts,v13-flat-obs

# Fast tags only, restricted scenario
./bench/run_suite.sh --tags v13-flat-obs --n 50000 --p 50

# XBART mode
./bench/run_suite.sh --mode xbart --tags gfr-v9-histogram,gfr-v10-flat-node-range

# Guard against slow early versions
./bench/run_suite.sh --tags v1-naive-cpp --n 5000 --timeout 120

# Tiny grid for testing the script itself
./bench/run_suite.sh --quick
```

| Option | Default | Description |
|---|---|---|
| `--tags TAG[,...]` | current checkout | Git tags to benchmark via worktrees |
| `--n N[,N,...]` | `10000,50000` | Training-set sizes |
| `--p P[,P,...]` | `10,50` | Feature counts |
| `--mode` | `bart` | `bart` or `xbart` |
| `--trees` | `200` | Trees in ensemble |
| `--burnin` | `200`/`15` | Burn-in iterations |
| `--samples` | `1000`/`25` | Posterior samples |
| `--timeout SECS` | `0` (none) | Kill a run after N seconds |
| `--results FILE` | `bench/results/bench_results.csv` | Output CSV |
| `--no-skip` | — | Re-run even if result already recorded |
| `--quick` | — | Tiny grid for smoke-testing |

### Multi-iteration sweep — `bench/run_suite_mc.sh`

Like `run_suite.sh`, but runs each configuration `--iters` times with independent random seeds, writing one row per iteration to `bench_results_mc.csv`. Use this when you want to average timing and RMSE over multiple data realisations for a more reliable comparison.

```bash
# Current checkout, 10 iterations at default scenario (n=50k, p=50)
./bench/run_suite_mc.sh

# Specific tags
./bench/run_suite_mc.sh --tags v12-leaf-counts,v13-flat-obs --iters 20

# Slower tags — restrict n to avoid very long runtimes
./bench/run_suite_mc.sh --tags v1-naive-cpp --n 10000 --timeout 300

# Tiny run for testing
./bench/run_suite_mc.sh --quick
```

Same options as `run_suite.sh`, plus:

| Option | Default | Description |
|---|---|---|
| `--iters N` | `10` | Independent repetitions per configuration |
| `--n N[,N,...]` | `50000` | Training-set sizes (narrower default than single-run) |
| `--results FILE` | `bench/results/bench_results_mc.csv` | Output CSV (C++ only) |

> **Note on slow tags:** v1–v4 (BART) and gfr-v1 (GFR) can each take several minutes per run at n=50k. Multiply by `--iters` before committing to a full sweep.

---

## Results files

| File | Written by | Content |
|---|---|---|
| `bench/results/bench_results.csv` | `run_suite.sh` | One row per (tag, n, p) run |
| `bench/results/bench_results_mc.csv` | `run_suite_mc.sh`, C++ binary | One row per iteration (C++ only) |
| `bench/results/bench_results_r.csv` | `bench_stochtree_r.R`, `bench_r_packages.R` | One row per iteration (R packages) |

## Viewing results

```bash
# Single-run CSV (default)
./bench/show_results.sh

# Multi-iteration CSV (aggregated — shows avg_ms and avg_rmse per version)
./bench/show_results.sh --mc

# Explicit file
./bench/show_results.sh bench/results/bench_results_mc.csv

# Filter options (work for both formats)
./bench/show_results.sh --mode xbart
./bench/show_results.sh --n 50000
./bench/show_results.sh --tag v13
```

The script auto-detects the CSV format from the header. Multi-iteration results are averaged across iterations before display.

---

## Plotting results

```bash
# From the single-run CSV (uses min time per version)
python bench/plot_results.py

# From the multi-iteration CSV (uses mean time and mean RMSE per version)
python bench/plot_results.py --mc

# Explicit CSV path
python bench/plot_results.py --csv bench/results/bench_results_mc.csv

# Display interactively instead of saving
python bench/plot_results.py --show

# Different scenario
python bench/plot_results.py --n 200000 --p 10
```

Produces four PNGs in `bench/results/`:

| File | Content |
|---|---|
| `speedup_bart.png` | BART MCMC wall-time by version |
| `speedup_gfr.png` | GFR/XBART wall-time by version |
| `rmse_bart.png` | BART MCMC test RMSE by version |
| `rmse_gfr.png` | GFR/XBART test RMSE by version |

The format is auto-detected from the CSV header; `--mc` is a shorthand for pointing at `bench_results_mc.csv`.

---

## Cross-implementation comparisons

All comparison scripts use the same DGP (`y = sin(2π·x₁) + N(0, σ²)`) and run `--iters` independent repetitions, reporting per-iteration timing and RMSE then an average. RNG implementations differ across languages, so individual data realisations will not match, but the DGP structure is identical.

Results from the R scripts are written to **`bench/results/bench_results_r.csv`** (shared by both scripts, separate from the C++ CSVs). Pass `--csv FILE` to override.

### stochtree R — `bench/bench_stochtree_r.R`

Times `stochtree::bart()` in MCMC or GFR mode.

```bash
Rscript bench/bench_stochtree_r.R
Rscript bench/bench_stochtree_r.R --mode xbart
Rscript bench/bench_stochtree_r.R --n_train 10000 --p 10 --iters 5
```

### dbarts, BART, and flexBART — `bench/bench_r_packages.R`

Times `dbarts`, `BART`, and `flexBART` implementations of the BART model back-to-back on the same data each iteration (MCMC only; none of these packages offer GFR).

```bash
Rscript bench/bench_r_packages.R
Rscript bench/bench_r_packages.R --n_train 10000 --p 10 --iters 5
```

### bartz (JAX/GPU) — `bench/bench_bartz.ipynb`

Jupyter notebook intended for Colab GPU. Set `n_iters` in the configuration cell. The first iteration includes JAX JIT compilation; the summary reports averages both with and without it.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrewherren/faststochtree/blob/main/bench/bench_bartz.ipynb)

---

## Tests

```bash
# Via ctest
ctest --test-dir build-release --output-on-failure

# Directly (verbose)
./build-release/test_bart
```
