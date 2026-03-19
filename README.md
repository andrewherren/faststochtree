# Fast Stochastic Forest Algorithms

Inspired by Giacomo Petrillo's jax-based [bartz](https://github.com/bartz-org/bartz) GPU-accelerated BART project, `faststochtree` provides an optimized / streamlined implementation of the BART MCMC and grow-from-root (GFR) algorithms, with a special focus on Apple Silicon hardware.

## Why Apple Silicon?

M-series machines are (a) high-performance consumer devices, and (b) very common in tech and academia. A BART implementation that makes efficient use of this platform would allow for fast model development / inference without cloud-based Nvidia GPU servers.

## Why a new project?

[stochtree](https://github.com/StochasticTree/stochtree) was built to be a robust library for flexible and composable BART modeling. However, its expressiveness means a large surface of supported features and models, which affords fewer opportunities for performance optimization.

Starting with the simplest possible BART model provides an ideal testbed for fine-tuning performance.
