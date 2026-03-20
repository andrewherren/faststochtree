#pragma once
#include "faststochtree/tree.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/thread_pool.hpp"
#include <cstring>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <cmath>

namespace bart {

struct BARTConfig {
    int   num_trees        = 200;
    int   tree_depth       = 6;      // max leaf depth; tree has 2^depth leaf slots
    int   min_samples_leaf = 5;
    float alpha            = 0.95f;  // BART tree prior: P(split at depth d) = alpha/(1+d)^beta
    float beta             = 2.0f;
    float leaf_prior_var   = 1.0f;   // tau: leaf value prior is N(0, tau)
    float sigma2_shape     = 3.0f;   // nu:     IG(nu/2, nu*lambda/2) prior on sigma2
    float sigma2_scale     = 1.0f;   // lambda
    int   p_eval           = 0;      // features evaluated per node: 0 = all p (gfr-v5)
    int   num_threads      = 1;      // thread pool size for GFR parallelism (gfr-v6/v7)
};

// Pre-allocated scratch workspace — one per BARTState, reused every sweep.
// Eliminates per-call heap allocations in propose_move and sample_leaves.
struct Workspace {
    // K=4 multilane scatter buffers for sample_leaves (depth=6: size 128)
    float s0[128], s1[128], s2[128], s3[128];
    int   c0[128], c1[128], c2[128], c3[128];
    // Scratch for tree.leaves() / tree.leaf_parents() — avoid per-call alloc
    std::vector<int> leaves_buf;
    std::vector<int> leaf_parents_buf;
    // All-zeros array of size n — used as pred_off=0 in gfr_sweep after
    // explicit restore (avoids a separate sample_leaves overload).
    std::vector<float> zeros;
};

// Persistent workspace for gfr-v9 histogram-based GFR.
// Replace the O(n*p) presorted copies with:
//   - A single flat_obs list (unordered, partitioned in-place)
//   - Per-node 256-bin sum/count histograms for all m features
//
// Per-node cost: O(n_k * m) histogram build + O(256 * m) prefix scan +
//                O(n_k) single obs-list partition.  No sorted-order copy needed.
struct GFRHistWorkspace {
    std::vector<int>     flat_obs;        // [n] obs indices, partitioned in-place
    // node_range[k] = {beg,end} for active node k; beg==-1 means inactive.
    // Flat vector indexed by node id (bounded by full_size) replaces unordered_map.
    std::vector<std::pair<int,int>> node_range;

    std::vector<float>   sum_hists;       // [m_max * 256]: fi*256 + bin
    std::vector<int>     cnt_hists;       // [m_max * 256]
    std::vector<float>   feat_log_total;  // [m_max + 1]
    std::vector<float>   cut_log_wts;     // stage-2 candidate log-weights (≤255)
    std::vector<uint8_t> cut_thresh_buf;  // stage-2 candidate thresholds  (≤255)
    std::vector<int>     right_buf;       // partition scratch
    std::vector<int>     feat_order;      // [p] Fisher-Yates scratch

    void alloc(int n, int p, int full_size) {
        flat_obs.resize(n);
        node_range.assign(full_size + 1, {-1, -1});
        sum_hists.resize(p * 256);
        cnt_hists.resize(p * 256);
        feat_log_total.resize(p + 1);
        cut_log_wts.reserve(255);
        cut_thresh_buf.reserve(255);
        right_buf.reserve(n);
        feat_order.resize(p);
    }

    void reinit(int n) {
        std::iota(flat_obs.begin(), flat_obs.end(), 0);
        std::fill(node_range.begin(), node_range.end(), std::make_pair(-1, -1));
        node_range[1] = {0, n};
    }
};

struct BARTState {
    int n, p;
    QuantizedX   Xq;    // quantized covariates (owned)
    const float* y;     // outcomes [n]

    std::vector<Tree>               trees;
    std::vector<std::vector<float>> pred;          // pred[t][i]: tree t's prediction for obs i
    std::vector<std::vector<int>>   leaf_indices;  // leaf_indices[t][i]: cached leaf node for obs i
    std::vector<std::vector<int>>   leaf_counts;   // leaf_counts[t][k]: # obs at node k in tree t
    std::vector<std::vector<int>>   flat_obs;      // flat_obs[t]: obs indices grouped by leaf (contiguous segments)
    std::vector<std::vector<int>>   leaf_start;    // leaf_start[t][k]: start index of leaf k in flat_obs[t]
    std::vector<float>              residual;      // partial residual
    float                           sigma2;
    Workspace                       ws;
    GFRHistWorkspace                gfr_hist_ws;     // persistent GFR workspace (gfr-v9)
    std::unique_ptr<ThreadPool>     thread_pool;     // optional; built by run_xbart for num_threads>1
};

// Reduced log marginal likelihood for a Gaussian constant leaf.
// Cancels terms that are equal between split and no-split comparisons.
inline float leaf_log_ml(float sum_y, int n, float sigma2, float tau) {
    return -0.5f * std::log(1.0f + tau * n / sigma2)
           + (tau * sum_y * sum_y) / (2.0f * sigma2 * (n * tau + sigma2));
}

// Sample sigma2 ~ IG(nu/2 + n/2,  nu*lambda/2 + RSS/2)
void sample_sigma2(const float* resid, int n, float& sigma2,
                   const BARTConfig& cfg, RNG& rng);

// Sample leaf values for all leaves of one tree (Gaussian posterior).
// Uses pre-cached leaf_idx[i] instead of re-traversing.
// pred_off[i] is added to resid[i] to form the effective partial residual.
// ws provides pre-allocated scatter lane buffers (zeroed on entry).
void sample_leaves(Tree& tree, const float* resid, const float* pred_off,
                   int n, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const std::vector<int>& leaf_idx, Workspace& ws);

} // namespace bart
