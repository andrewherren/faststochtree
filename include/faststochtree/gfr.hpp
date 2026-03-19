#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"

namespace bart {

// Grow tree from root using GFR (grow-from-root / XBART).
// Resets tree, then samples a new tree structure via BFS + softmax-weighted
// split selection. v2: uses PresortedX for O(n*p) per rebuild vs O(n*p*log n).
void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps);

// One full GFR sweep: rebuild all T trees + sample sigma2.
void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

} // namespace bart
