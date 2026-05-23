#import "MetalContext.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

// ---------------------------------------------------------------------------
// MSL source — compiled at runtime via newLibraryWithSource:.
// Embedding source keeps the build system simple for now; we'll switch to
// offline .metallib compilation once the kernels stabilise.
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// --- noop (dispatch-latency probe) ----------------------------------------

kernel void noop(uint id                  [[thread_position_in_grid]],
                 device volatile uint* out [[buffer(0)]])
{
    if (id == 0) out[0] = id;
}

// --- histogram_build -------------------------------------------------------
//
// One threadgroup per (feature, node) pair.
// Grid: threadgroups.x = p (features),  threadgroups.y = n_nodes
// Threads per threadgroup: HIST_TG_SIZE (must evenly divide 256).
//
// Each threadgroup:
//   1. Zeroes 256-bin threadgroup-local sum/count histograms.
//   2. Strides over obs_list[beg..end), scatters into local histograms.
//   3. Writes the local histograms to global output buffers.
//
// Float atomics: Metal has no native float atomic-add on threadgroup memory,
// so we use a CAS loop over the uint bit-cast — the standard GPU workaround.
// Contention is low (256 bins, uniformly distributed observations), so the
// loop rarely retries more than once.
//
// Output layout: [node * p * 256 + feat * 256 + bin]

constant uint HIST_TG_SIZE = 64;  // must divide 256 evenly (4 bins per thread)

// Shared helper used by scan_feat_log_total.
inline float leaf_log_ml(float sum_y, float n, float sigma2, float tau) {
    return -0.5f * log(1.0f + tau * n / sigma2)
           + (tau * sum_y * sum_y) / (2.0f * sigma2 * (n * tau + sigma2));
}

inline void tg_atomic_add_float(threadgroup atomic_uint* addr, float val) {
    uint cur = atomic_load_explicit(addr, memory_order_relaxed);
    uint next;
    do {
        next = as_type<uint>(as_type<float>(cur) + val);
    } while (!atomic_compare_exchange_weak_explicit(
                addr, &cur, next,
                memory_order_relaxed, memory_order_relaxed));
}

kernel void histogram_build(
    device const uint8_t* Xq           [[buffer(0)]],  // [n * p] col-major
    device const float*   residuals    [[buffer(1)]],  // [n]
    device const int*     obs_list     [[buffer(2)]],  // [sum of node n_k values]
    device const int*     node_ranges  [[buffer(3)]],  // [n_nodes * 2]: {beg, end}
    device       float*   sum_hists    [[buffer(4)]],  // [n_nodes * m * 256]
    device       int*     cnt_hists    [[buffer(5)]],  // [n_nodes * m * 256]
    constant     int&     n_total      [[buffer(6)]],  // total obs in dataset
    constant     int&     m_total      [[buffer(7)]],  // features evaluated per node (≤ p)
    device const int*     feat_order   [[buffer(8)]],  // [n_nodes * m_total]; nil when m == p
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    const uint fi   = tg_pos.x;  // feature slot in [0, m_total)
    const uint node = tg_pos.y;

    // Map fi → actual feature column (identity when feat_order is nil).
    const uint feat_actual = (feat_order != nullptr)
        ? (uint)feat_order[node * (uint)m_total + fi]
        : fi;

    const int beg = node_ranges[node * 2];
    const int end = node_ranges[node * 2 + 1];

    // Threadgroup-local histograms: sum stored as float bits, count as uint.
    threadgroup atomic_uint sum_bits[256];
    threadgroup atomic_uint cnt[256];

    // Phase 1: zero local histograms (4 bins per thread when tg_size = 64).
    const uint bpt = 256u / HIST_TG_SIZE;
    const uint b0  = tid * bpt;
    for (uint b = b0; b < b0 + bpt; b++) {
        atomic_store_explicit(&sum_bits[b], 0u, memory_order_relaxed);
        atomic_store_explicit(&cnt[b],      0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: scatter over observations in this node.
    const device uint8_t* col = Xq + (int)feat_actual * n_total;
    for (int k = beg + (int)tid; k < end; k += (int)HIST_TG_SIZE) {
        const int   obs = obs_list[k];
        const uint  bin = col[obs];
        const float r   = residuals[obs];
        tg_atomic_add_float(&sum_bits[bin], r);
        atomic_fetch_add_explicit(&cnt[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: write local histogram to global memory.
    const uint out_base = (node * (uint)m_total + fi) * 256u;
    for (uint b = b0; b < b0 + bpt; b++) {
        sum_hists[out_base + b] = as_type<float>(
            atomic_load_explicit(&sum_bits[b], memory_order_relaxed));
        cnt_hists[out_base + b] = (int)
            atomic_load_explicit(&cnt[b], memory_order_relaxed);
    }
}
// --- scan_feat_log_total -------------------------------------------------------
//
// Second compute pass, run in the same MTLCommandBuffer as histogram_build.
// The implicit GPU barrier between compute encoders guarantees histogram writes
// are visible before this kernel reads them — no explicit fence needed.
//
// One thread per (feature, node): walks bins 0..254 of that feature's histogram
// in order, accumulates sum_L/count_L as a prefix scan, and computes the
// online log-sum-exp over all valid split points.
//
// Grid: threadgroups = (p, n_nodes, 1), threadsPerThreadgroup = (1, 1, 1).
// The GPU packs independent single-thread threadgroups into SIMD groups
// automatically, achieving parallelism across (feat, node) pairs.
//
// Output: feat_log[node * p + feat]

// --- select_split -------------------------------------------------------------
//
// One thread per active node.  Reads feat_log (output of scan_feat_log_total)
// and the raw sum/count histograms to sample (feature, cutpoint) from the GFR
// split posterior using two-stage softmax.
//
// Stage 1: softmax over m feature-slots + 1 no-split option.
// Stage 2: softmax over valid cutpoints in the chosen feature's histogram.
//
// Uses xorshift32 RNG seeded per node by the CPU; state must be nonzero.
//
// Output layout: out_feat[ai] = actual column index (or -1 for leaf),
//                out_thresh[ai] = quantile bin (uint8).

inline uint xorshift32(thread uint& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

inline float xor_uniform(thread uint& state) {
    return float(xorshift32(state)) * (1.0f / 4294967296.0f);
}

kernel void select_split(
    device const float*   feat_log        [[buffer(0)]],   // [n_active * m]
    device const float*   sum_hists       [[buffer(1)]],   // [n_active * m * 256]
    device const int*     cnt_hists       [[buffer(2)]],   // [n_active * m * 256]
    device const float*   log_split_ratio [[buffer(3)]],   // [n_active]
    device const uint*    rng_seeds       [[buffer(4)]],   // [n_active], nonzero
    device       int*     out_feat        [[buffer(5)]],   // [n_active]: actual col or -1
    device       uint8_t* out_thresh      [[buffer(6)]],   // [n_active]: chosen bin
    constant     float&   sigma2          [[buffer(7)]],
    constant     float&   tau             [[buffer(8)]],
    constant     int&     min_leaf        [[buffer(9)]],
    constant     int&     m_total         [[buffer(10)]],
    device const int*     feat_order      [[buffer(11)]],  // [n_active * m] or nil
    uint ai [[thread_position_in_grid]]
) {
    uint rng = rng_seeds[ai];
    const int m = m_total;

    // sum_T / count_T from slot-0 histogram (all slots have same marginal total).
    const device float* slot0_sum = sum_hists + (int)ai * m * 256;
    const device int*   slot0_cnt = cnt_hists  + (int)ai * m * 256;
    float sum_T = 0.f; int count_T = 0;
    for (int b = 0; b < 256; b++) { sum_T += slot0_sum[b]; count_T += slot0_cnt[b]; }

    const device float* node_fl = feat_log + (int)ai * m;

    int n_valid = 0;
    for (int fi = 0; fi < m; fi++)
        if (node_fl[fi] > -INFINITY) n_valid++;

    float lsr = log_split_ratio[ai];
    float no_split_lw = leaf_log_ml(sum_T, float(count_T), sigma2, tau)
                        - lsr
                        + (n_valid > 0 ? log(float(n_valid)) : 0.f);

    // Stage 1: softmax over (feat_log[0..m-1], no_split_lw).
    float max_ft = no_split_lw;
    for (int fi = 0; fi < m; fi++)
        if (node_fl[fi] > max_ft) max_ft = node_fl[fi];

    float total_ft = exp(no_split_lw - max_ft);
    for (int fi = 0; fi < m; fi++)
        if (node_fl[fi] > -INFINITY) total_ft += exp(node_fl[fi] - max_ft);

    float u1  = xor_uniform(rng);
    float cum1 = 0.f;
    int chosen_fi = m;  // default = no-split
    for (int fi = 0; fi < m; fi++) {
        if (node_fl[fi] > -INFINITY) cum1 += exp(node_fl[fi] - max_ft);
        if (u1 * total_ft <= cum1) { chosen_fi = fi; break; }
    }

    if (chosen_fi == m) {
        out_feat[ai]   = -1;
        out_thresh[ai] = 0;
        return;
    }

    int chosen_feat = (feat_order != nullptr) ? feat_order[(int)ai * m + chosen_fi] : chosen_fi;

    // Stage 2: cutpoint softmax over bins 0..254 of the chosen feature slot.
    const device float* sh = sum_hists + ((int)ai * m + chosen_fi) * 256;
    const device int*   ch = cnt_hists  + ((int)ai * m + chosen_fi) * 256;

    // Pass 1: compute log-sum-exp over valid splits for normalisation.
    float sum_L = 0.f; int count_L = 0;
    float max_lw2 = -INFINITY, lse2 = 0.f;
    int n_valid_cuts = 0;
    uint8_t last_thresh = 0;

    for (int b = 0; b < 255; b++) {
        if (ch[b] == 0) continue;
        sum_L += sh[b]; count_L += ch[b];
        int count_R = count_T - count_L;
        if (count_L < min_leaf || count_R < min_leaf) continue;
        float lw = leaf_log_ml(sum_L,          float(count_L), sigma2, tau)
                 + leaf_log_ml(sum_T - sum_L,  float(count_R), sigma2, tau);
        if (lw > max_lw2) { lse2 = lse2 * exp(max_lw2 - lw) + 1.f; max_lw2 = lw; }
        else                lse2 += exp(lw - max_lw2);
        n_valid_cuts++;
        last_thresh = (uint8_t)b;
    }

    if (n_valid_cuts == 0) {
        out_feat[ai]   = -1;
        out_thresh[ai] = 0;
        return;
    }

    // Pass 2: CDF walk; default to last valid cut (matches CPU fallback).
    float u2  = xor_uniform(rng);
    float target2 = u2 * lse2;
    sum_L = 0.f; count_L = 0;
    float cum2 = 0.f;
    uint8_t chosen_thresh = last_thresh;
    bool found = false;

    for (int b = 0; b < 255; b++) {
        if (ch[b] == 0) continue;
        sum_L += sh[b]; count_L += ch[b];
        int count_R = count_T - count_L;
        if (count_L < min_leaf || count_R < min_leaf) continue;
        float lw = leaf_log_ml(sum_L,          float(count_L), sigma2, tau)
                 + leaf_log_ml(sum_T - sum_L,  float(count_R), sigma2, tau);
        cum2 += exp(lw - max_lw2);
        if (!found && target2 <= cum2) {
            chosen_thresh = (uint8_t)b;
            found = true;
        }
    }

    out_feat[ai]   = chosen_feat;
    out_thresh[ai] = chosen_thresh;
}

// --- scan_feat_log_total -------------------------------------------------------

kernel void scan_feat_log_total(
    device const float* sum_hists [[buffer(0)]],  // [n_nodes * m * 256]
    device const int*   cnt_hists [[buffer(1)]],  // [n_nodes * m * 256]
    device       float* feat_log  [[buffer(2)]],  // [n_nodes * m]
    constant     float& sigma2    [[buffer(3)]],
    constant     float& tau       [[buffer(4)]],
    constant     int&   min_leaf  [[buffer(5)]],
    constant     int&   m_total   [[buffer(6)]],  // features evaluated per node
    uint2 tg_pos [[threadgroup_position_in_grid]]
) {
    const uint fi   = tg_pos.x;  // feature slot in [0, m_total)
    const uint node = tg_pos.y;

    const uint base = (node * (uint)m_total + fi) * 256u;
    const device float* sh = sum_hists + base;
    const device int*   ch = cnt_hists + base;

    // Compute node totals from this feature slot's histogram.
    float sum_T = 0.f; int count_T = 0;
    for (uint b = 0u; b < 256u; b++) { sum_T += sh[b]; count_T += ch[b]; }

    // Prefix scan with online log-sum-exp over valid split points.
    float sum_L = 0.f; int count_L = 0;
    float max_lw = -INFINITY, lse = 0.f;

    for (uint b = 0u; b < 255u; b++) {
        if (ch[b] == 0) continue;
        sum_L   += sh[b]; count_L += ch[b];
        int count_R = count_T - count_L;
        if (count_L < min_leaf || count_R < min_leaf) continue;
        float lml = leaf_log_ml(sum_L,          float(count_L), sigma2, tau)
                  + leaf_log_ml(sum_T - sum_L,  float(count_R), sigma2, tau);
        if (lml > max_lw) { lse = lse * exp(max_lw - lml) + 1.f; max_lw = lml; }
        else                 lse += exp(lml - max_lw);
    }

    feat_log[node * (uint)m_total + fi] = (lse > 0.f) ? (max_lw + log(lse)) : -INFINITY;
}
)MSL";
// ---------------------------------------------------------------------------

namespace gpu {

struct MetalContext::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLLibrary>              library  = nil;
    id<MTLComputePipelineState> noop_pso   = nil;
    id<MTLComputePipelineState> hist_pso   = nil;
    id<MTLComputePipelineState> scan_pso   = nil;
    id<MTLComputePipelineState> select_pso = nil;
    std::string                 name;

    // Persistent shared buffers — grown on demand, never shrunk.
    // Eliminates per-dispatch Metal buffer allocation for feat_order and node_ranges.
    id<MTLBuffer> buf_fo     = nil;  NSUInteger buf_fo_cap     = 0;
    id<MTLBuffer> buf_ranges = nil;  NSUInteger buf_ranges_cap = 0;

    id<MTLBuffer> ensure(id<MTLBuffer>& buf, NSUInteger& cap,
                         const void* src, NSUInteger bytes) {
        if (bytes > cap) {
            buf = [device newBufferWithLength:bytes
                          options:MTLResourceStorageModeShared];
            cap = bytes;
        }
        memcpy([buf contents], src, bytes);
        return buf;
    }
};

MetalContext::MetalContext() : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        fprintf(stderr, "[Metal] No Metal-capable device found.\n");
        return;
    }

    impl_->name  = [[impl_->device name] UTF8String];
    impl_->queue = [impl_->device newCommandQueue];

    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kKernelSrc];
    impl_->library = [impl_->device newLibraryWithSource:src options:nil error:&err];
    if (!impl_->library) {
        fprintf(stderr, "[Metal] Shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }

    auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn = [impl_->library
            newFunctionWithName:[NSString stringWithUTF8String:name]];
        if (!fn) {
            fprintf(stderr, "[Metal] Function '%s' not found in library.\n", name);
            return nil;
        }
        id<MTLComputePipelineState> pso =
            [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!pso)
            fprintf(stderr, "[Metal] PSO '%s' error: %s\n",
                    name, [[err localizedDescription] UTF8String]);
        return pso;
    };

    impl_->noop_pso   = make_pso("noop");
    impl_->hist_pso   = make_pso("histogram_build");
    impl_->scan_pso   = make_pso("scan_feat_log_total");
    impl_->select_pso = make_pso("select_split");
}

MetalContext::~MetalContext() { delete impl_; }

bool MetalContext::ok() const {
    return impl_ && impl_->device && impl_->noop_pso && impl_->hist_pso
        && impl_->scan_pso && impl_->select_pso;
}

const char* MetalContext::device_name() const {
    return impl_ ? impl_->name.c_str() : "(no device)";
}

double MetalContext::noop_roundtrip_us() {
    if (!impl_ || !impl_->noop_pso) return -1.0;

    id<MTLBuffer> out = [impl_->device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    auto host_t0 = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer>         cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->noop_pso];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    auto host_t1 = std::chrono::high_resolution_clock::now();

    double gpu_us  = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    double host_us = std::chrono::duration<double, std::micro>(host_t1 - host_t0).count();

    printf("  noop: gpu=%.1f µs  host_roundtrip=%.1f µs\n", gpu_us, host_us);
    return gpu_us;
}

MetalContext::HistResult MetalContext::histogram_build(
    const uint8_t* Xq, const float* resid,
    const int* obs_list, int obs_count, const int* node_ranges,
    int n, int p, int n_nodes,
    int m, const int* feat_order,
    float* out_sum, int* out_cnt)
{
    HistResult res;
    if (!impl_ || !impl_->hist_pso) return res;

    auto new_shared = [&](size_t bytes) {
        return [impl_->device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
    };
    auto fill_shared = [&](const void* src, size_t bytes) {
        id<MTLBuffer> b = [impl_->device newBufferWithBytes:src
                               length:bytes
                               options:MTLResourceStorageModeShared];
        return b;
    };

    id<MTLBuffer> buf_Xq    = fill_shared(Xq,       (size_t)n * p       * sizeof(uint8_t));
    id<MTLBuffer> buf_resid = fill_shared(resid,     (size_t)n           * sizeof(float));
    id<MTLBuffer> buf_obs   = fill_shared(obs_list,  (size_t)obs_count   * sizeof(int));
    id<MTLBuffer> buf_ranges = impl_->ensure(impl_->buf_ranges, impl_->buf_ranges_cap,
                                             node_ranges, (size_t)n_nodes * 2 * sizeof(int));
    id<MTLBuffer> buf_fo = feat_order
        ? impl_->ensure(impl_->buf_fo, impl_->buf_fo_cap,
                        feat_order, (size_t)n_nodes * m * sizeof(int))
        : nil;

    const int out_elems = n_nodes * m * 256;
    id<MTLBuffer> buf_sum = new_shared((size_t)out_elems * sizeof(float));
    id<MTLBuffer> buf_cnt = new_shared((size_t)out_elems * sizeof(int));

    // ---- begin timed region ----
    auto host_t0 = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer>         cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->hist_pso];
    [enc setBuffer:buf_Xq     offset:0 atIndex:0];
    [enc setBuffer:buf_resid  offset:0 atIndex:1];
    [enc setBuffer:buf_obs    offset:0 atIndex:2];
    [enc setBuffer:buf_ranges offset:0 atIndex:3];
    [enc setBuffer:buf_sum    offset:0 atIndex:4];
    [enc setBuffer:buf_cnt    offset:0 atIndex:5];
    [enc setBytes:&n          length:sizeof(int) atIndex:6];
    [enc setBytes:&m          length:sizeof(int) atIndex:7];
    [enc setBuffer:buf_fo     offset:0 atIndex:8];

    // Grid: (m feature slots) x (n_nodes) threadgroups.
    [enc dispatchThreadgroups:MTLSizeMake(m, n_nodes, 1)
       threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    auto host_t1 = std::chrono::high_resolution_clock::now();
    // ---- end timed region ----

    res.gpu_kernel_us = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    res.host_us = std::chrono::duration<double, std::micro>(host_t1 - host_t0).count();

    if (out_sum)
        memcpy(out_sum, [buf_sum contents], (size_t)out_elems * sizeof(float));
    if (out_cnt)
        memcpy(out_cnt, [buf_cnt contents], (size_t)out_elems * sizeof(int));

    return res;
}

MetalContext::HistResult MetalContext::histogram_and_scan(
    const uint8_t* Xq, const float* resid,
    const int* obs_list, int obs_count, const int* node_ranges,
    int n, int p, int n_nodes,
    int m, const int* feat_order,
    float sigma2, float tau, int min_samples_leaf,
    float* out_sum, int* out_cnt, float* out_feat_log)
{
    HistResult res;
    if (!impl_ || !impl_->hist_pso || !impl_->scan_pso) return res;

    auto new_shared = [&](size_t bytes) {
        return [impl_->device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
    };
    auto fill_shared = [&](const void* src, size_t bytes) {
        return [impl_->device newBufferWithBytes:src length:bytes
                               options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_Xq    = fill_shared(Xq,      (size_t)n * p       * sizeof(uint8_t));
    id<MTLBuffer> buf_resid = fill_shared(resid,    (size_t)n           * sizeof(float));
    id<MTLBuffer> buf_obs   = fill_shared(obs_list, (size_t)obs_count   * sizeof(int));
    id<MTLBuffer> buf_ranges = impl_->ensure(impl_->buf_ranges, impl_->buf_ranges_cap,
                                             node_ranges, (size_t)n_nodes * 2 * sizeof(int));
    id<MTLBuffer> buf_fo = feat_order
        ? impl_->ensure(impl_->buf_fo, impl_->buf_fo_cap,
                        feat_order, (size_t)n_nodes * m * sizeof(int))
        : nil;

    const int hist_elems = n_nodes * m * 256;
    const int log_elems  = n_nodes * m;

    id<MTLBuffer> buf_sum = new_shared((size_t)hist_elems * sizeof(float));
    id<MTLBuffer> buf_cnt = new_shared((size_t)hist_elems * sizeof(int));
    id<MTLBuffer> buf_log = new_shared((size_t)log_elems  * sizeof(float));

    // ---- begin timed region ----
    auto host_t0 = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer> cmd = [impl_->queue commandBuffer];

    // Pass 1: histogram_build
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->hist_pso];
        [enc setBuffer:buf_Xq     offset:0 atIndex:0];
        [enc setBuffer:buf_resid  offset:0 atIndex:1];
        [enc setBuffer:buf_obs    offset:0 atIndex:2];
        [enc setBuffer:buf_ranges offset:0 atIndex:3];
        [enc setBuffer:buf_sum    offset:0 atIndex:4];
        [enc setBuffer:buf_cnt    offset:0 atIndex:5];
        [enc setBytes:&n          length:sizeof(int) atIndex:6];
        [enc setBytes:&m          length:sizeof(int) atIndex:7];
        [enc setBuffer:buf_fo     offset:0 atIndex:8];
        [enc dispatchThreadgroups:MTLSizeMake(m, n_nodes, 1)
           threadsPerThreadgroup:MTLSizeMake(64, 1, 1)];
        [enc endEncoding];  // implicit GPU barrier: scan pass waits for histogram writes
    }

    // Pass 2: scan_feat_log_total (reads buf_sum/buf_cnt written by pass 1)
    {
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:impl_->scan_pso];
        [enc setBuffer:buf_sum    offset:0 atIndex:0];
        [enc setBuffer:buf_cnt    offset:0 atIndex:1];
        [enc setBuffer:buf_log    offset:0 atIndex:2];
        [enc setBytes:&sigma2          length:sizeof(float) atIndex:3];
        [enc setBytes:&tau             length:sizeof(float) atIndex:4];
        [enc setBytes:&min_samples_leaf length:sizeof(int)  atIndex:5];
        [enc setBytes:&m               length:sizeof(int)   atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(m, n_nodes, 1)
           threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [enc endEncoding];
    }

    [cmd commit];
    [cmd waitUntilCompleted];

    auto host_t1 = std::chrono::high_resolution_clock::now();
    // ---- end timed region ----

    res.gpu_kernel_us = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    res.host_us = std::chrono::duration<double, std::micro>(host_t1 - host_t0).count();

    if (out_sum)      memcpy(out_sum,      [buf_sum contents], (size_t)hist_elems * sizeof(float));
    if (out_cnt)      memcpy(out_cnt,      [buf_cnt contents], (size_t)hist_elems * sizeof(int));
    if (out_feat_log) memcpy(out_feat_log, [buf_log contents], (size_t)log_elems  * sizeof(float));

    return res;
}

void MetalContext::select_splits(const float* feat_log,
                                 const float* sum_hists, const int* cnt_hists,
                                 const float* log_split_ratio,
                                 const unsigned* rng_seeds,
                                 int n_active, int m,
                                 float sigma2, float tau, int min_leaf,
                                 const int* feat_order,
                                 SplitResult* out)
{
    if (!impl_ || !impl_->select_pso || n_active == 0) return;

    auto fill_shared = [&](const void* src, size_t bytes) {
        return [impl_->device newBufferWithBytes:src length:bytes
                               options:MTLResourceStorageModeShared];
    };
    auto new_shared = [&](size_t bytes) {
        return [impl_->device newBufferWithLength:bytes
                               options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_fl  = fill_shared(feat_log,        (size_t)n_active * m       * sizeof(float));
    id<MTLBuffer> buf_sum = fill_shared(sum_hists,       (size_t)n_active * m * 256 * sizeof(float));
    id<MTLBuffer> buf_cnt = fill_shared(cnt_hists,       (size_t)n_active * m * 256 * sizeof(int));
    id<MTLBuffer> buf_lsr = fill_shared(log_split_ratio, (size_t)n_active           * sizeof(float));
    id<MTLBuffer> buf_rng = fill_shared(rng_seeds,       (size_t)n_active           * sizeof(unsigned));
    id<MTLBuffer> buf_ofe = new_shared( (size_t)n_active                            * sizeof(int));
    id<MTLBuffer> buf_oth = new_shared( (size_t)n_active                            * sizeof(uint8_t));
    id<MTLBuffer> buf_fo  = feat_order
        ? fill_shared(feat_order, (size_t)n_active * m * sizeof(int))
        : nil;

    id<MTLCommandBuffer>         cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->select_pso];
    [enc setBuffer:buf_fl  offset:0 atIndex:0];
    [enc setBuffer:buf_sum offset:0 atIndex:1];
    [enc setBuffer:buf_cnt offset:0 atIndex:2];
    [enc setBuffer:buf_lsr offset:0 atIndex:3];
    [enc setBuffer:buf_rng offset:0 atIndex:4];
    [enc setBuffer:buf_ofe offset:0 atIndex:5];
    [enc setBuffer:buf_oth offset:0 atIndex:6];
    [enc setBytes:&sigma2   length:sizeof(float) atIndex:7];
    [enc setBytes:&tau      length:sizeof(float) atIndex:8];
    [enc setBytes:&min_leaf length:sizeof(int)   atIndex:9];
    [enc setBytes:&m        length:sizeof(int)   atIndex:10];
    [enc setBuffer:buf_fo  offset:0 atIndex:11];
    [enc dispatchThreads:MTLSizeMake((NSUInteger)n_active, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    const int*     raw_feat   = static_cast<const int*>    ([buf_ofe contents]);
    const uint8_t* raw_thresh = static_cast<const uint8_t*>([buf_oth contents]);
    for (int ai = 0; ai < n_active; ai++) {
        out[ai].feat   = raw_feat[ai];
        out[ai].thresh = raw_thresh[ai];
    }
}

} // namespace gpu
