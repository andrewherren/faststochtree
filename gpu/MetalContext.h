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

    // Build p x 256 sum/count histograms for n_nodes nodes in one dispatch.
    //
    // Xq          [n * p]         column-major uint8 quantized covariates
    // resid       [n]             partial residuals
    // obs_list    [obs_count]     observation indices; ranges index into this array
    // obs_count                   number of valid elements in obs_list
    // node_ranges [n_nodes * 2]   {beg, end} pairs into obs_list, one per node
    // out_sum     [n_nodes*p*256] sum histograms — written on return
    // out_cnt     [n_nodes*p*256] count histograms — written on return
    //
    // Output layout: out_sum[(node * p + feat) * 256 + bin]
    // Single-node convenience: pass node_ranges = {0, n_k} and n_nodes = 1.
    // obs_count must cover the maximum end index in node_ranges.
    HistResult histogram_build(const uint8_t* Xq, const float* resid,
                               const int* obs_list, int obs_count,
                               const int* node_ranges,
                               int n, int p, int n_nodes,
                               float* out_sum, int* out_cnt);

    // Two-pass variant: histogram_build followed by scan_feat_log_total in one
    // MTLCommandBuffer. The implicit GPU barrier between compute encoders
    // guarantees histogram writes are visible to the scan pass.
    //
    // sigma2, tau, min_samples_leaf: GFR split-scoring parameters for the scan.
    // out_feat_log [n_nodes * p]: log-sum-exp over valid split log-weights per
    //   (node, feature) pair. Layout: out_feat_log[node * p + feat].
    // gpu_kernel_us covers both passes combined.
    HistResult histogram_and_scan(const uint8_t* Xq, const float* resid,
                                  const int* obs_list, int obs_count,
                                  const int* node_ranges,
                                  int n, int p, int n_nodes,
                                  float sigma2, float tau, int min_samples_leaf,
                                  float* out_sum, int* out_cnt,
                                  float* out_feat_log);

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gpu
