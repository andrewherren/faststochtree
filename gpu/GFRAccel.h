#pragma once
#include "faststochtree/model.hpp"
#include "faststochtree/rng.hpp"
#include "faststochtree/sampler.hpp"
#include "MetalContext.h"

namespace gpu {

// GPU-accelerated grow_tree_gfr.
// Supports cfg.p_eval feature subsampling (Fisher-Yates per node before each dispatch).
// Restriction: z == nullptr (constant leaf only).
// Each BFS level dispatches all active nodes as one batched Metal command buffer.
// Prefix scan / stage1 / stage2 / partition run on CPU after each dispatch.
void grow_tree_gfr_gpu(bart::Tree& tree, const bart::QuantizedX& Xq,
                       const float* resid, int n, int p, float sigma2,
                       const bart::BARTConfig& cfg, bart::RNG& rng,
                       bart::GFRHistWorkspace& ws, MetalContext& gpu_ctx);

// Drop-in for bart::gfr_sweep (single-forest, constant leaf, no thread pool).
void gfr_sweep_gpu(bart::BARTState& state, const bart::BARTConfig& cfg,
                   bart::RNG& rng, MetalContext& gpu_ctx);

// High-level fit functions — same signature as bart::fit_xbart / bart::fit_warmstart_bart
// but run all GFR sweeps on the GPU. Constructs MetalContext internally.
// Throws std::runtime_error if Metal is unavailable (non-macOS build).
// Restriction: constant leaf only (no z). Supports cfg.p_eval feature subsampling.
bart::BARTModel fit_xbart_gpu(const float* X, const float* y, int n, int p,
                               const float* X_test, int n_test,
                               const bart::BARTConfig& cfg,
                               int n_burnin, int n_samples, int seed);

bart::BARTModel fit_warmstart_bart_gpu(const float* X, const float* y, int n, int p,
                                        const float* X_test, int n_test,
                                        const bart::BARTConfig& cfg,
                                        int n_gfr_burnin, int n_mcmc_burnin,
                                        int n_samples, int seed,
                                        bool keep_gfr_samples = false);

} // namespace gpu
