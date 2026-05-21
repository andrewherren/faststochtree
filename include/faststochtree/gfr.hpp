#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/thread_pool.hpp"
#include <vector>

namespace bart {

// Grow tree from root using GFR (grow-from-root / XBART).
// v10: flat node_range array + 256-bin histograms + single obs-list partition.
//   - node_range is a flat vector<pair<int,int>> (no hash overhead).
//   - O(n_k*m) histogram build + O(256*m) prefix scan per node.
//   - O(n_k) partition update (single obs list).
//   - ws is a persistent workspace reused across calls (no per-call malloc).
// pool=nullptr → single-threaded.
// z=nullptr → constant leaf (standard BART/XBART).
// z!=nullptr → regression leaf on z; ws.zwt_hists must be allocated (has_z=true).
void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   GFRHistWorkspace& ws, ThreadPool* pool = nullptr,
                   const float* z = nullptr);

// One full GFR sweep: rebuild all T trees + sample sigma2.
void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

// One BCF GFR sweep over both forests sharing a single residual and sigma2.
// tau_state.gfr_hist_ws must be allocated with has_z=true.
// tau_state.pred[t][i] stores β_leaf(i)·z_i on exit.
void bcf_gfr_sweep(std::vector<float>& shared_resid, const float* z,
                   BARTState& mu_state, const BARTConfig& mu_cfg,
                   BARTState& tau_state, const BARTConfig& tau_cfg,
                   float& sigma2, RNG& rng);

} // namespace bart
