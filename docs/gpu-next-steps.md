# GPU Acceleration for `faststochtree`: Paths Forward

The original goal of this project was simple: `bartz`, but for the apple silicon GPU. I tried using `mlx` and quickly hit a wall with the lack of control flow primitives (like `jax`'s [`while_loop`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html) on which bartz relies). Moving down to a pure C++ codebase, the next step was to try to make rich use of the CPU. I used claude to fine tune the CPU performance of minimalist BART / XBART implementations and wrote about it [here](https://andrewherren.github.io/posts/2026-04-23-bart-laptop/). Since then, I've extended the CPU implementation to BCF / XBCF for causal inference (see [here](https://github.com/andrewherren/faststochtree/blob/2db469100d751d756040f21c27c2b852e7dc6788/R/faststochtree.R#L174) for the code, though some proper documentation is forthcoming).

What's next? Figuring out the GPU story!

## Overview

In the BART MCMC algorithm, we perform several steps over and over again for each tree:

1. Determine the mapping from every observation to its corresponding leaf of each tree
2. Update the residual to remove the contribution of a given tree
3. Propose a modification to the tree - grow or prune - and compute sufficient statistics of the impacted leaves
4. Compute log integrated likelihoods and accept or reject proposal
5. Compute sufficient statistics for all leaves of the newly modified tree and sample posterior parameters
6. Update the residual to include the contribution of the newly sampled tree

Fast CPU-based implementations typically maintain in-memory data structures to avoid unnecessary passes through the data in steps 1 and 5. GPUs' extreme parallelism changes the nature of this tradeoff: simple passes through the data are "cheap" but coordinating modifications to complicated data structures is often trickier. `bartz` splits these steps into

* work that can be done in parallel across every tree (namely, 1 and some of the work behind 3, 4, and 5), and 
* work that must be done sequentially (anything involving the partial residual),

and implements them in a `jax` program that expresses this parallel-sequential split. `jax` compiles the steps into efficient CUDA kernels through [XLA](https://openxla.org/).

To make `faststochtree` work on the GPU, we cannot rely on `jax` and instead must write GPU kernels in a way that works with our C++ codebase and targets our platform of interest. In practice, this means writing in a C++ inspired dialect like [CUDA](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) for NVIDIA GPUs, [ROCm](https://rocm.docs.amd.com/en/latest/) for AMD GPUs, or [Metal](https://developer.apple.com/metal/) for Apple Silicon GPUs.

## Initial Experiments

Translating the steps outlined above into GPU code is subtle. Does each step get its own kernel? Does some work get merged into a single kernel? Is there an advantage to re-ordering operations as in `bartz`'s design?

Initial attempts to port the BART to Metal for the Apple Silicon GPU hit several walls:

1. GPU kernel launch overhead: if we express the steps above as separate kernels, the overhead of launching multiple kernels per tree per MCMC iteration is substantial
2. Lack of grid synchronization primitives: if we try to fuse the entire MCMC sampler into a single kernel, there are several steps within each iteration when we must ensure that every thread of a GPU kernel has completed its work, and Metal doesn't provide native barriers for this ([CUDA does](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html)); we can program our own barrier ([Xiao and Feng (2010)](https://synergy.cs.vt.edu/pubs/papers/xiao-ipdps2010-gpusync.pdf)), but it has a lot of overhead and fails when we have many threadgroups (i.e., large data)

The [`experimental/gpu`](https://github.com/andrewherren/faststochtree/tree/main/experimental/gpu) folder of this repo includes minimal, self-contained programs that validate these limitations:

1. Metal ([`experimental/gpu/metal`](https://github.com/andrewherren/faststochtree/tree/main/experimental/gpu/metal), built with a simple `make`): `dispatch_overhead` measures the ~25 µs cost of a kernel launch, and `grid_barrier` shows the wall we hit when we try to directly implement a software barrier
2. CUDA ([`experimental/gpu/cuda/reprex.ipynb`](https://github.com/andrewherren/faststochtree/tree/main/experimental/gpu/cuda)): a Colab notebook that compiles and runs CUDA benchmarks on a T4 (free on Colab); profiles both the cost of a kernel launch (naive and CUDA-Graph replay), and the cost of CUDA's `grid.sync()` barrier

## Next Steps

| Scenario | Description | Verdict |
| --- | ----------- | ----------- |
| CUDA GPU-Only | Write CUDA kernel(s) to run MCMC algorithm end-to-end | Promising! |
| CUDA mixed CPU-GPU | Run MCMC algorithm as combination of CUDA kernels and CPU (host) code | Likely non-starter due to data transfer costs |
| Silicon GPU-Only | Write Metal kernel(s) to run MCMC algorithm end-to-end | Non-starter due to kernel launch and grid sync overhead |
| Silicon mixed CPU-GPU | Run MCMC algorithm as combination of Metal kernels and CPU code | Could work; initial experimentation validates idea for large `n` |

The most obvious next step is to extend `faststochtree` with CUDA, likely as a fully GPU-resident kernel that runs every iteration of the sampler on the GPU without incurring the CPU to GPU data transfer penalty. Another underexplored avenue is to use a mix of CPU and GPU on Apple Silicon, which could work because of Apple unified memory architecture (the CPU and GPU share the same address space).
