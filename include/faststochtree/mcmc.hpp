#pragma once
#include "faststochtree/model.hpp"
#include <vector>

namespace bart {

// Initialize BARTState: allocate trees and pred buffers, set residual = y, sigma2 = 1
void init_state(BARTState& state, const BARTConfig& cfg, RNG& rng);

// One MCMC sweep: for each tree, propose grow/prune, sample leaves, update sigma2
void mcmc_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

// Regression leaf sampler for the BCF τ forest (basis ω_i = z_i).
// Accumulates s_wyx = Σ r_i·z_i and s_wxx = Σ z_i per leaf.
void sample_leaves_regression(Tree& tree, const float* resid, const float* pred_off,
                               const float* z, int n, float sigma2, const BARTConfig& cfg,
                               RNG& rng, const std::vector<int>& leaf_idx, Workspace& ws);

// One BCF MCMC sweep over both forests sharing a single residual and sigma2.
// mu_state and tau_state must already be initialised via init_state().
// shared_resid = y − μ_forest − τ_forest·z on entry and exit.
// tau_state.pred[t][i] stores β_leaf(i)·z_i (not the raw leaf value).
void bcf_mcmc_sweep(std::vector<float>& shared_resid, const float* z,
                    BARTState& mu_state, const BARTConfig& mu_cfg,
                    BARTState& tau_state, const BARTConfig& tau_cfg,
                    float& sigma2, RNG& rng);

} // namespace bart
