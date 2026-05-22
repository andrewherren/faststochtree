# faststochtree GPU Backend: A Reading Guide

A practical walkthrough of the Metal GPU code for someone who wants to
understand, modify, or extend it. Assumes basic C++ familiarity but no
prior GPU experience.

---

## 1. Why GPU, and why Metal?

The GFR (grow-from-root) inner loop spends most of its time building
256-bin histograms — one per feature per node. At p=100 features, this
is 100 independent histogram builds per node, each scanning through the
node's observations. Those 100 builds have no data dependencies on each
other, which makes them embarrassingly parallel and a natural GPU target.

Metal is Apple's GPU compute API (macOS/iOS). On Apple Silicon the CPU
and GPU share the same physical RAM ("unified memory"), which eliminates
the DMA copies that dominate GPU overhead on discrete cards. This makes
Metal viable even for workloads with moderate data sizes.

---

## 2. File map

```
gpu/
  MetalContext.h      Pure C++ header. No ObjC types visible — callers
                      only need this file, not Metal headers.
  MetalContext.mm     ObjC++ implementation. Owns the Metal device,
                      command queue, compiled kernels, and dispatch logic.
  GFRAccel.h          C++ header for the GFR integration layer.
  GFRAccel.cpp        Restructured BFS loop that dispatches GPU per level.

bench/
  gpu_bench.cpp       Standalone benchmarks: noop roundtrip, single-node
                      histogram sweep, BFS-level sweep, GFR sweep comparison.
```

The split between `.h`/`.mm` is deliberate: `.mm` is Objective-C++
(needed to call Metal APIs), but the header is plain C++, so the rest of
the codebase doesn't need to know Metal exists.

---

## 3. Metal API concepts used here

You'll encounter these types throughout the code.

| Type | What it is |
|---|---|
| `MTLDevice` | The GPU. Created once via `MTLCreateSystemDefaultDevice()`. |
| `MTLCommandQueue` | A serial queue of work submitted to the GPU. One per context. |
| `MTLLibrary` | A compiled collection of GPU functions (kernels). We compile from embedded MSL source at startup. |
| `MTLComputePipelineState` (PSO) | A compiled, ready-to-dispatch version of one kernel. Think of it as a function pointer for the GPU. |
| `MTLCommandBuffer` | One unit of work submitted to the queue. Can contain multiple sequential compute passes. |
| `MTLComputeCommandEncoder` | Encodes one compute pass into a command buffer: set kernel, set buffers, dispatch threads. |
| `MTLBuffer` | A region of GPU-accessible memory. With `StorageModeShared` on Apple Silicon, this is the same physical RAM the CPU uses — zero copy. |

The lifecycle is always: encode → commit → wait.

```objc
id<MTLCommandBuffer> cmd = [queue commandBuffer];
id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
// ... set PSO, buffers, dispatch ...
[enc endEncoding];
[cmd commit];
[cmd waitUntilCompleted];  // blocks CPU until GPU finishes
```

---

## 4. The kernels (MSL)

All kernel source lives in the `kKernelSrc` string in `MetalContext.mm`,
compiled at runtime via `newLibraryWithSource:`. The language is Metal
Shading Language (MSL), which is essentially C++14 with GPU-specific
qualifiers.

### 4a. `histogram_build`

**What it does:** For each `(feature, node)` pair, counts observations
into 256 bins and accumulates residual sums per bin.

**Grid layout:**
```
threadgroups:          (p, n_nodes, 1)   — one TG per (feat, node)
threadsPerThreadgroup: (64, 1, 1)        — 64 threads cooperate per TG
```

Each threadgroup handles exactly one `(feature, node)` pair. The 64
threads within it cooperate using *threadgroup-local* (shared) memory.

**Three phases inside each threadgroup:**

```
Phase 1 — zero:    Each thread zeroes 4 bins of the shared histogram
                   (256 bins / 64 threads = 4 bins per thread).
                   threadgroup_barrier() — all threads must finish before phase 2.

Phase 2 — scatter: Each thread strides through obs_list[beg..end) with
                   step=64, loading obs index → column value → residual,
                   then atomically adds to the shared histogram.
                   threadgroup_barrier() — wait for all scatters.

Phase 3 — write:   Each thread writes its 4 bins from shared memory to
                   the global output buffer.
```

**Float atomics:** Metal has no native float atomic-add on threadgroup
memory. The workaround is a CAS (compare-and-swap) loop on the uint
bit-cast of the float. Contention is low (256 bins, ~uniform
distribution), so the loop rarely retries.

```metal
inline void tg_atomic_add_float(threadgroup atomic_uint* addr, float val) {
    uint cur = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    do {
        next = as_type<uint>(as_type<float>(cur) + val);
    } while (!atomic_compare_exchange_weak_explicit(
                addr, &cur, next, memory_order_relaxed, memory_order_relaxed));
}
```

**Output layout:** `sum_hists[(node * p + feat) * 256 + bin]`

### 4b. `scan_feat_log_total`

**What it does:** For each `(feature, node)` pair, reads the histogram
written by pass 1 and computes the log-sum-exp over all valid split
log-marginal-likelihoods. The result is one float per `(feat, node)`:
the `feat_log_total[fi]` value that GFR uses for the feature softmax.

**Grid layout:**
```
threadgroups:          (p, n_nodes, 1)   — one TG per (feat, node)
threadsPerThreadgroup: (1, 1, 1)         — one thread per TG
```

One thread walks the 256 bins sequentially (a prefix scan), accumulating
`sum_L` / `count_L` as it goes. For each valid split point (both sides
meet `min_samples_leaf`), it evaluates `leaf_log_ml` for left and right
and folds the result into an online log-sum-exp.

**Online log-sum-exp:** Numerically stable accumulation without storing
all values:
```
if lml > max_lw:  lse = lse * exp(max_lw - lml) + 1;  max_lw = lml
else:             lse += exp(lml - max_lw)
result = max_lw + log(lse)
```

**Why 1 thread per threadgroup?** The scan is inherently sequential (each
bin depends on the previous prefix sum), so there's no parallelism *within*
a feature. Parallelism comes from running many `(feat, node)` threadgroups
concurrently. The GPU scheduler packs 32 single-thread threadgroups into
one SIMD group, so utilisation is reasonable even though each threadgroup
is tiny.

**Output layout:** `feat_log[node * p + feat]`

---

## 5. The two-pass design

`histogram_and_scan` encodes both kernels into **one `MTLCommandBuffer`**
with two sequential encoders:

```objc
id<MTLCommandBuffer> cmd = [queue commandBuffer];

// Pass 1
id<MTLComputeCommandEncoder> enc1 = [cmd computeCommandEncoder];
// ... encode histogram_build ...
[enc1 endEncoding];    // ← implicit GPU barrier here

// Pass 2
id<MTLComputeCommandEncoder> enc2 = [cmd computeCommandEncoder];
// ... encode scan_feat_log_total, reading buf_sum/buf_cnt from pass 1 ...
[enc2 endEncoding];

[cmd commit];
[cmd waitUntilCompleted];
```

The key line is `[enc1 endEncoding]`. Metal guarantees that all writes
from a compute encoder are visible to the next encoder in the same command
buffer before it starts. This is the "implicit GPU barrier" — no explicit
fence or synchronisation primitive needed. Both passes share the same
`buf_sum` and `buf_cnt` MTLBuffers: pass 1 writes them, pass 2 reads them.

The single `[cmd waitUntilCompleted]` covers both passes. GPU timestamps
(`cmd.GPUStartTime` / `cmd.GPUEndTime`) span both passes.

**Why not two separate command buffers?** You'd pay the ~200µs dispatch
roundtrip twice. Fusing into one buffer costs one roundtrip for both
passes combined.

---

## 6. GFRAccel: the integration layer

`grow_tree_gfr_gpu` is a restructured version of `grow_tree_gfr` with
one key difference: at each BFS level, all active nodes are dispatched
together in one GPU command buffer rather than processed sequentially.

**Per-level flow:**

```
1. Scan current_level, separate nodes into:
     - immediate leaves (too small / max depth) → ws.leaf_segs
     - active (eligible to split) → active[]

2. Build node_ranges_buf: flat array of {beg, end} pairs for active nodes.

3. GPU dispatch (histogram_and_scan):
     - Grid: (p × n_active) threadgroups for histogram
     - Grid: (p × n_active) threadgroups for scan
     - Outputs: gpu_sum[n_active * p * 256]
                gpu_cnt[n_active * p * 256]
                gpu_feat_log[n_active * p]

4. For each active node (CPU):
     a. Read sum_T / count_T from feature 0's histogram slice (256 adds)
     b. Copy feat_log_total[0..p-1] from gpu_feat_log (p copies)
     c. Add no-split option feat_log_total[p] (leaf_log_ml on CPU)
     d. Stage 1 softmax → chosen feature (O(p))
     e. Stage 2 cutpoint scan over chosen feature's histogram (O(256))
     f. tree.grow() + obs-list partition (O(n_k), scalar, CPU)

5. Push children to next_level.
```

**Why does partition stay on CPU?** The partition result (modified
`ws.flat_obs`) feeds the next level's GPU dispatch. Moving it to GPU
would require a separate kernel with a parallel prefix scan + scatter,
plus an extra dispatch roundtrip per level. At n=100k the CPU partition
is ~1–2ms total per tree — real but not dominant.

**Restrictions vs. the CPU version:**
- `cfg.p_eval` must be 0 (no feature subsampling). The GPU dispatch
  always builds histograms for all p features. Supporting subsampling
  would require sending the shuffled feature order to the GPU.
- `z == nullptr` only (constant leaf). The BCF regression-leaf path uses
  `zwt_hists` which the GPU kernel doesn't currently compute.

---

## 7. Memory layout quick reference

| Buffer | Layout | Size |
|---|---|---|
| `Xq` (input) | column-major: `data[feat * n + obs]` | n × p bytes |
| `resid` (input) | flat: `resid[obs]` | n × 4 bytes |
| `obs_list` (input) | flat: `obs_list[k]` = obs index | obs_count × 4 bytes |
| `node_ranges` (input) | `{beg, end}` pairs | n_nodes × 8 bytes |
| `sum_hists` (output) | `[(node * p + feat) * 256 + bin]` | n_nodes × p × 256 × 4 bytes |
| `cnt_hists` (output) | same layout | n_nodes × p × 256 × 4 bytes |
| `feat_log` (output) | `[node * p + feat]` | n_nodes × p × 4 bytes |

The column-major layout of `Xq` is important: `feat * n + obs` means
all observations for one feature are contiguous, so the histogram kernel's
inner loop (`col[obs_list[k]]`) has sequential memory access within a
feature — good for cache on both CPU and GPU.

---

## 8. Timing model

Two timing numbers are reported for each dispatch:

- **`gpu_kernel_us`**: GPU hardware time, from `cmd.GPUStartTime` to
  `cmd.GPUEndTime`. This is what the GPU actually spent executing kernels.
  It excludes encode time and any queue latency.

- **`host_us`**: wall-clock time from just before `[cmd commit]` to just
  after `[cmd waitUntilCompleted]`. This includes encode + submit + GPU
  execution + notification overhead — the full latency the caller experiences.

On M1 Max, a minimal noop kernel shows ~9µs GPU time and ~190–220µs host
time. That ~200µs gap is the irreducible dispatch overhead: command buffer
submission, GPU scheduling, and CPU wakeup latency. It's paid once per
`MTLCommandBuffer`, regardless of how much work is inside — which is why
fusing multiple passes into one buffer matters.

---

## 9. Building

```bash
# Configure with GPU support
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DGPU_METAL=ON

# Build just the GPU bench
cmake --build build-gpu --target gpu_bench -j4

# Run
./build-gpu/gpu/gpu_bench
```

The GPU target is off by default (`GPU_METAL=OFF`) so the rest of the
build is unaffected. The `gpu/` subdirectory is only added when the
option is on.

---

## 10. Where to go next

**To add a new GPU kernel:**
1. Write the MSL function in the `kKernelSrc` string in `MetalContext.mm`.
2. Add a PSO member to `MetalContext::Impl` and compile it in the
   constructor via `make_pso("your_function_name")`.
3. Add a method to `MetalContext` (declared in `.h`, defined in `.mm`)
   that encodes and dispatches it.
4. Call the new method from `GFRAccel.cpp` or wherever appropriate.

**To add GPU partition (the next big step):**
The algorithm is: mark (obs ≤ thresh ? 1 : 0) → parallel prefix scan →
scatter. The prefix scan is the key primitive. See the resources section
of the companion document for references. `flat_obs` would need to become
a persistent `MTLBuffer` rather than a CPU vector passed anew each level.

**To support feature subsampling (p_eval < p):**
The Fisher-Yates shuffle produces a `feat_order[]` permutation. You'd
pass this as a buffer to the histogram kernel and have each threadgroup
look up its actual feature index via `feat_order[tg_pos.x]` rather than
using `tg_pos.x` directly as the feature. The scan kernel would need the
same permutation to write `feat_log` in the correct `fi` slot.
