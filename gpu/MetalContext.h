#pragma once
#include <cstdint>

namespace gpu {

// RAII wrapper around MTLDevice + MTLCommandQueue.
// Compiles Metal kernels from embedded MSL source on construction.
// All ObjC types are hidden in Impl — this header is pure C++.
struct MetalContext {
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    // False if no Metal-capable device was found.
    bool ok() const;

    // Human-readable device name (e.g. "Apple M3 Pro").
    const char* device_name() const;

    // Dispatch a no-op kernel; returns GPU kernel time in microseconds.
    // Prints both GPU kernel time and host-side roundtrip time.
    double noop_roundtrip_us();

    // Timing result returned by histogram_build.
    struct HistResult {
        double gpu_kernel_us = -1.0;  // GPU hardware time for the kernel only
        double host_us       = -1.0;  // host wall-clock (encode+submit+wait)
    };

    // Build p x 256 sum/count histograms for one node.
    //
    // Xq       [n * p] column-major uint8 quantized covariates
    // resid    [n]     partial residuals
    // obs_list [n_k]   observation indices belonging to this node
    // out_sum  [p * 256]  sum of resid[obs] per (feature, bin) — written on return
    // out_cnt  [p * 256]  count of obs per (feature, bin)      — written on return
    //
    // Output layout: out_sum[feat * 256 + bin]
    HistResult histogram_build(const uint8_t* Xq, const float* resid,
                               const int* obs_list,
                               int n, int p, int n_k,
                               float* out_sum, int* out_cnt);

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gpu
