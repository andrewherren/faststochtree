# `experimental/gpu` — minimal reprexes for the GPU story

These programs back the claims in [`docs/gpu-next-steps.md`](../../docs/gpu-next-steps.md) with code that can be compiled and run. They isolate the two GPU limitations that decide whether a fully GPU-resident BART sampler is feasible on a given platform:

1. **Kernel launch / dispatch overhead**: how expensive is one tiny kernel?
2. **Device-wide ("grid") synchronization**: can every thread cheaply rendezvous between phases?

The BART MCMC algorithm needs device-wide barriers at least once per tree in every iteration of the algorithm. With 200 trees and 1000 MCMC iterations, this is 200k barriers, so the cost of this synchronization adds up quickly if it is expensive. Similarly, if we use kernels to enforce this synchronization, then the cost of the kernel becomes salient.

## Layout

| Folder | Platform | How to run | Reprexes |
| --- | --- | --- | --- |
| [`metal/`](metal/) | Apple Silicon (Metal) | `make` (see its README) | `dispatch_overhead`, `grid_barrier` |
| [`cuda/`](cuda/)   | NVIDIA (CUDA)         | open `reprex.ipynb` in Colab, set runtime to GPU | launch overhead, grid barrier |

The two examples are an attempt at measuring the same phenomena in both Metal and CUDA to demonstrate that Metal has some limitations that make it challenging for accelerating the BART MCMC algorithm.

## The result in one table

| | Metal (M1 Max, measured) | CUDA (T4, measured) |
| --- | --- | --- |
| per-launch / per-dispatch | ~25 µs/command buffer; no graph replay across a sync | naive ~10 µs/launch; **CUDA Graph replay ~1.6 µs** |
| device-wide barrier | software only (Xiao & Feng 2010); ~0.6 µs at 1–2 TG, **cliffs to tens of ms / livelocks** | hardware `grid.sync()` **~1.7–2.1 µs, ~flat** (1→160 blocks) |
