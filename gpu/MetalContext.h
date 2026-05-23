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

    // Build m x 256 sum/count histograms for n_nodes nodes in one dispatch.
    //
    // Xq          [n * p]         column-major uint8 quantized covariates
    // resid       [n]             partial residuals
    // obs_list    [obs_count]     observation indices; ranges index into this array
    // obs_count                   number of valid elements in obs_list
    // node_ranges [n_nodes * 2]   {beg, end} pairs into obs_list, one per node
    // m                           features evaluated per node (≤ p)
    // feat_order  [n_nodes * m]   per-node feature permutation; nullptr when m == p
    // out_sum     [n_nodes*m*256] sum histograms — written on return
    // out_cnt     [n_nodes*m*256] count histograms — written on return
    //
    // Output layout: out_sum[(node * m + feat_slot) * 256 + bin]
    HistResult histogram_build(const uint8_t* Xq, const float* resid,
                               const int* obs_list, int obs_count,
                               const int* node_ranges,
                               int n, int p, int n_nodes,
                               int m, const int* feat_order,
                               float* out_sum, int* out_cnt);

    // Two-pass variant: histogram_build followed by scan_feat_log_total in one
    // MTLCommandBuffer. The implicit GPU barrier between compute encoders
    // guarantees histogram writes are visible to the scan pass.
    //
    // m, feat_order: same semantics as histogram_build above.
    // sigma2, tau, min_samples_leaf: GFR split-scoring parameters for the scan.
    // out_feat_log [n_nodes * m]: log-sum-exp over valid split log-weights per
    //   (node, feature-slot) pair. Layout: out_feat_log[node * m + feat_slot].
    // gpu_kernel_us covers both passes combined.
    HistResult histogram_and_scan(const uint8_t* Xq, const float* resid,
                                  const int* obs_list, int obs_count,
                                  const int* node_ranges,
                                  int n, int p, int n_nodes,
                                  int m, const int* feat_order,
                                  float sigma2, float tau, int min_samples_leaf,
                                  float* out_sum, int* out_cnt,
                                  float* out_feat_log);

    // Split sampling decision for one active node.
    struct SplitResult {
        int     feat;    // -1 = no-split (node becomes a leaf)
        uint8_t thresh;  // quantile bin for the chosen cutpoint
    };

    // Three-pass fused dispatch: histogram_build → scan_feat_log_total → select_split
    // in a single MTLCommandBuffer (one commit per BFS level instead of two).
    //
    // Inputs/semantics match histogram_and_scan + select_splits combined.
    // log_split_ratio [n_nodes] and rng_seeds [n_nodes] are the per-node
    // inputs needed by the select_split pass.
    // out [n_nodes]: written split decisions (feat=-1 means no-split/leaf).
    void histogram_scan_select(const uint8_t* Xq, const float* resid,
                               const int* obs_list, int obs_count,
                               const int* node_ranges,
                               int n, int p, int n_nodes, int m,
                               const int* feat_order,
                               float sigma2, float tau, int min_samples_leaf,
                               const float* log_split_ratio,
                               const unsigned* rng_seeds,
                               SplitResult* out);

    // For each active node (0..n_active-1), sample (feat, thresh) on the GPU
    // from the GFR two-stage split posterior.
    //
    // feat_log        [n_active * m]       output of scan_feat_log_total
    // sum_hists       [n_active * m * 256] raw sum histograms
    // cnt_hists       [n_active * m * 256] raw count histograms
    // log_split_ratio [n_active]           log(p_split / (1-p_split)) per node
    // rng_seeds       [n_active]           nonzero xorshift32 seeds (one per node)
    // feat_order      [n_active * m] or nullptr when m == p
    // out             [n_active]           written split decisions
    void select_splits(const float* feat_log,
                       const float* sum_hists, const int* cnt_hists,
                       const float* log_split_ratio,
                       const unsigned* rng_seeds,
                       int n_active, int m,
                       float sigma2, float tau, int min_leaf,
                       const int* feat_order,
                       SplitResult* out);

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gpu
