#pragma once
#include "faststochtree/tree.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/rng.hpp"
#include <cstring>
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

// Pre-sorted observation indices — argsort by each feature, built once.
// Reused across all tree rebuilds in GFR/XBART sweeps.
struct PresortedX {
    std::vector<std::vector<int>> sorted_idx;  // [p][n]: sorted obs indices per feature
    void build(const QuantizedX& Xq, int n, int p);
};

struct BARTState {
    int n, p;
    QuantizedX   Xq;    // quantized covariates (owned)
    const float* y;     // outcomes [n]

    std::vector<Tree>               trees;
    std::vector<std::vector<float>> pred;          // pred[t][i]: tree t's prediction for obs i
    std::vector<std::vector<int>>   leaf_indices;  // leaf_indices[t][i]: cached leaf node for obs i
    std::vector<float>              residual;      // partial residual
    float                           sigma2;
    Workspace                       ws;
    PresortedX                      presorted;     // built by init_gfr; empty for MCMC-only use
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
