#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/thread_pool.hpp"

namespace bart {

// Grow tree from root using GFR (grow-from-root / XBART).
// v7: level-BFS, feature-parallel scan + partition update via thread pool.
// v8: part is a persistent workspace reused across calls (no O(n*p) malloc).
// pool=nullptr → single-threaded.
void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps, TreePartition& part, ThreadPool* pool = nullptr);

// One full GFR sweep: rebuild all T trees + sample sigma2.
void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

} // namespace bart
