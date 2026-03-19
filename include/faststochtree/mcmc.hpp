#pragma once
#include "faststochtree/model.hpp"

namespace bart {

// Initialize BARTState: allocate trees and pred buffers, set residual = y, sigma2 = 1
void init_state(BARTState& state, const BARTConfig& cfg, RNG& rng);

// One MCMC sweep: for each tree, propose grow/prune, sample leaves, update sigma2
void mcmc_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng);

} // namespace bart
