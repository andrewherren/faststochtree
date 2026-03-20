#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/thread_pool.hpp"

namespace bart {

// Grow tree from root using GFR (grow-from-root / XBART).
// v10: flat node_range array + 256-bin histograms + single obs-list partition.
//   - node_range is a flat vector<pair<int,int>> (no hash overhead).
//   - O(n_k*m) histogram build + O(256*m) prefix scan per node.
//   - O(n_k) partition update (single obs list).
//   - ws is a persistent workspace reused across calls (no per-call malloc).
// pool=nullptr → single-threaded.
void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   GFRHistWorkspace& ws, ThreadPool* pool = nullptr);

// One full GFR sweep: rebuild all T trees + sample sigma2.
void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

} // namespace bart
