# Metal reprexes — launch overhead & the missing grid barrier

Two minimal, self-contained Metal programs that demonstrate the two limitations
that wall a fully GPU-resident BART sampler on Apple Silicon. Each `.mm` file
compiles its own shader at runtime (no `metallib` step) and prints a small table.

The CUDA counterparts live in `../cuda/reprex.ipynb` — run them on a borrowed
NVIDIA card (a free Colab T4 is enough) to see the other side of each comparison.

## Requirements

macOS with the Xcode command-line tools:

```sh
xcode-select --install
```

## Build & run

```sh
make            # builds both
make run        # builds, then runs both
# or individually:
./dispatch_overhead
./grid_barrier
```

## What each one shows

### `dispatch_overhead` — Metal launches are expensive and un-amortizable

Times an empty kernel. The figure that matters is **(b)**: one command buffer per
dispatch, ~**25 µs** to encode + commit on an M1 Max. Every phase that must observe a
device-wide synchronization is a separate command buffer, and a phase recurs several
times per tree, every tree, every sweep — so this ~25 µs constant is multiplied by a
large factor over a run. Metal has **no CUDA-Graph-style replay** that can cross a
device-wide sync, so it cannot be amortized. (Case (a), many dispatches in one buffer,
is cheap — but those dispatches can't synchronize across the whole grid.)

Measured (Apple M1 Max):

```
  (a) one cmd buffer, 2000 encoded dispatches :   ~2.6 us/dispatch (GPU)
  (b) 2000 separate command buffers           :  ~25.1 us/dispatch (host: encode+commit)
```

### `grid_barrier` — Metal has no hardware grid barrier, and the software one cliffs

Metal exposes no device-wide barrier (CUDA's `grid.sync()`). The only option is a
hand-rolled software barrier — the **Xiao & Feng (2010)** counter/generation
rendezvous spun on a device-atomic counter. It is cheap for a handful of
threadgroups, then **detonates**: once more threadgroups are launched than the
scheduler co-schedules for a spinning kernel, the spinners wait on groups that
aren't resident yet (a forward-progress hazard). Past that point a single barrier
goes from ~1 µs to tens of ms, and then *livelocks outright* — the dispatch never
returns. The program sweeps the threadgroup count upward and **stops at the first
cliff** (with a 15 s per-dispatch host timeout) so it always completes instead of
hanging.

A representative run (Apple M1 Max):

```
  N_TG           ms (GPU)       ns/barrier
  1                0.0308            615.0
  2                0.0283            565.0
  4         >15 s — barrier did not return (livelock)

  ^ CLIFF at 4 threadgroups: the software barrier livelocked outright.
```

**The cliff's exact location is non-deterministic** — across runs on the same
machine it lands anywhere from 4 to 16 threadgroups (in an earlier run
it sat at 8: ~1.6 µs/barrier for ≤4 threadgroups, then ~18.7 ms at 8 and ~44 ms at 16). That
run-to-run variability is itself the signature of scheduler livelock rather than
deterministic work. What is invariant: below the cliff a barrier costs ~0.5–1.6 µs;
at or above it, it costs tens of ms or never returns.

> Repeatedly killing and restarting these runs can leave orphaned GPU dispatches that
> jam the device (so even 1 threadgroup appears to livelock) until the watchdog
> (`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`) reaps them — give the GPU
> a few seconds to settle between runs.

Compare CUDA (`../cuda/reprex.ipynb`): `grid.sync()` is ~**1.7–2.1 µs/barrier and
~flat** — measured on a Colab T4 it drifts only from 1678 ns at 1 block to 2142 ns at
160 (the co-resident maximum), a 1.3x rise across 160x more blocks. It *stays* flat as
you add blocks instead of collapsing.

## The takeaway

A single GPU-resident BART chain has to synchronize across all observations to update
the residual at least once per tree — several device-wide barriers per tree in the
early Metal experiments — repeated for every tree on every sweep. On Metal each such
sync is either ~25 µs of command-buffer overhead (separate buffers) or a software
barrier that livelocks past a handful of threadgroups. Multiplied across trees and
sweeps, either wall sinks it. Both are absent on CUDA. This is the empirical basis for
the verdicts in [`../../../docs/gpu-next-steps.md`](../../../docs/gpu-next-steps.md).
