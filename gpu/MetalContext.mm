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
    device       float*   sum_hists    [[buffer(4)]],  // [n_nodes * p * 256]
    device       int*     cnt_hists    [[buffer(5)]],  // [n_nodes * p * 256]
    constant     int&     n_total      [[buffer(6)]],  // total obs in dataset
    constant     int&     p_total      [[buffer(7)]],  // total features
    uint2 tg_pos  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_index_in_threadgroup]]
) {
    const uint feat = tg_pos.x;
    const uint node = tg_pos.y;

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
    const device uint8_t* col = Xq + (int)feat * n_total;
    for (int k = beg + (int)tid; k < end; k += (int)HIST_TG_SIZE) {
        const int   obs = obs_list[k];
        const uint  bin = col[obs];
        const float r   = residuals[obs];
        tg_atomic_add_float(&sum_bits[bin], r);
        atomic_fetch_add_explicit(&cnt[bin], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: write local histogram to global memory.
    const uint out_base = (node * (uint)p_total + feat) * 256u;
    for (uint b = b0; b < b0 + bpt; b++) {
        sum_hists[out_base + b] = as_type<float>(
            atomic_load_explicit(&sum_bits[b], memory_order_relaxed));
        cnt_hists[out_base + b] = (int)
            atomic_load_explicit(&cnt[b], memory_order_relaxed);
    }
}
)MSL";
// ---------------------------------------------------------------------------

namespace gpu {

struct MetalContext::Impl {
    id<MTLDevice>               device   = nil;
    id<MTLCommandQueue>         queue    = nil;
    id<MTLLibrary>              library  = nil;
    id<MTLComputePipelineState> noop_pso = nil;
    id<MTLComputePipelineState> hist_pso = nil;
    std::string                 name;
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

    impl_->noop_pso = make_pso("noop");
    impl_->hist_pso = make_pso("histogram_build");
}

MetalContext::~MetalContext() { delete impl_; }

bool MetalContext::ok() const {
    return impl_ && impl_->device && impl_->noop_pso && impl_->hist_pso;
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
    float* out_sum, int* out_cnt)
{
    HistResult res;
    if (!impl_ || !impl_->hist_pso) return res;

    // All buffers use shared storage: no DMA copy on Apple Silicon.
    // Buffer creation is intentionally excluded from timing — in a full
    // integration these would be persistent GPU-resident buffers.
    auto shared_buf_from = [&](const void* src, size_t bytes) {
        return [impl_->device newBufferWithBytes:src
                               length:bytes
                               options:MTLResourceStorageModeShared];
    };

    id<MTLBuffer> buf_Xq     = shared_buf_from(Xq,          (size_t)n * p       * sizeof(uint8_t));
    id<MTLBuffer> buf_resid  = shared_buf_from(resid,        (size_t)n           * sizeof(float));
    id<MTLBuffer> buf_obs    = shared_buf_from(obs_list,     (size_t)obs_count   * sizeof(int));
    id<MTLBuffer> buf_ranges = shared_buf_from(node_ranges,  (size_t)n_nodes * 2 * sizeof(int));

    const int out_elems = n_nodes * p * 256;
    id<MTLBuffer> buf_sum = [impl_->device
        newBufferWithLength:(size_t)out_elems * sizeof(float)
        options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_cnt = [impl_->device
        newBufferWithLength:(size_t)out_elems * sizeof(int)
        options:MTLResourceStorageModeShared];

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
    [enc setBytes:&p          length:sizeof(int) atIndex:7];

    // Grid: (p features) x (n_nodes) threadgroups.
    [enc dispatchThreadgroups:MTLSizeMake(p, n_nodes, 1)
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

} // namespace gpu
