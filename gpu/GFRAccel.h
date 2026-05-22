#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "MetalContext.h"

namespace gpu {

// GPU-accelerated grow_tree_gfr.
// Restrictions: m == p (cfg.p_eval must be 0), z == nullptr (constant leaf only).
// Each BFS level dispatches all active nodes as one batched Metal command buffer.
// Prefix scan / stage1 / stage2 / partition run on CPU after each dispatch.
void grow_tree_gfr_gpu(bart::Tree& tree, const bart::QuantizedX& Xq,
                       const float* resid, int n, int p, float sigma2,
                       const bart::BARTConfig& cfg, bart::RNG& rng,
                       bart::GFRHistWorkspace& ws, MetalContext& gpu_ctx);

// Drop-in for bart::gfr_sweep (single-forest, constant leaf, no thread pool).
void gfr_sweep_gpu(bart::BARTState& state, const bart::BARTConfig& cfg,
                   bart::RNG& rng, MetalContext& gpu_ctx);

} // namespace gpu
