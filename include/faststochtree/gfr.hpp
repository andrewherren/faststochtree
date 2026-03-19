#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/thread_pool.hpp"

namespace bart {

// Grow tree from root using GFR (grow-from-root / XBART).
// v6: level-BFS with optional parallel node evaluation via thread pool.
// pool=nullptr → single-threaded.
void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps, ThreadPool* pool = nullptr);

// One full GFR sweep: rebuild all T trees + sample sigma2.
void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

} // namespace bart
