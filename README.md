# Fast Stochastic Forest Algorithms

Inspired by Giacomo Petrillo's jax-based [bartz](https://github.com/bartz-org/bartz) GPU-accelerated BART project, `faststochtree` provides an optimized / streamlined implementation of the BART MCMC and grow-from-root (GFR) algorithms, with a special focus on Apple Silicon hardware.

## Why Apple Silicon?

M-series machines are (a) high-performance consumer devices, and (b) very common in tech and academia. A BART implementation that makes efficient use of this platform would allow for fast model development / inference without cloud-based Nvidia GPU servers.

## Why a new project?

[stochtree](https://github.com/StochasticTree/stochtree) was built to be a robust library for flexible and composable BART modeling. However, its expressiveness means a large surface of supported features and models, which affords fewer opportunities for performance optimization.

Starting with the simplest possible BART model provides an ideal testbed for fine-tuning performance.

# Getting Started

Requires CMake ≥ 3.20 and a C++17 compiler. An internet connection is needed on first build to fetch GoogleTest.

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j$(sysctl -n hw.logicalcpu)
```

## Running the benchmark

```bash
./build-release/faststochtree_bench [options]
```

All options are optional; defaults match the standard benchmark fixture.

| Option | Default | Description |
|---|---|---|
| `--n_train` | `50000` | Training set size |
| `--n_test` | `500` | Test set size |
| `--p` | `50` | Number of covariates |
| `--trees` | `200` | Number of trees in the forest |
| `--burnin` | `200` | Number of burn-in sweeps (discarded) |
| `--samples` | `1000` | Number of posterior samples |
| `--seed` | `12345` | Random seed |
| `--sigma` | `1.0` | True noise standard deviation |

Example — quick run on a smaller dataset:

```bash
./build-release/faststochtree_bench --n_train 2000 --p 10 --trees 50 --burnin 50 --samples 100
```

## Running the tests

The test suite can be run via `ctest`

```bash
ctest --test-dir build-release --output-on-failure
```

Or by running the GoogleTest binary directly for more verbose output

```bash
./build-release/test_bart
```
