#pragma once
#include "faststochtree/tree.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/rng.hpp"
#include <vector>
#include <cmath>

namespace bart {

struct BARTConfig {
    int   num_trees        = 200;
    int   min_samples_leaf = 5;
    float alpha            = 0.95f;  // BART tree prior: P(split at depth d) = alpha/(1+d)^beta
    float beta             = 2.0f;
    float leaf_prior_var   = 1.0f;   // tau: leaf value prior is N(0, tau)
    float sigma2_shape     = 3.0f;   // nu:     IG(nu/2, nu*lambda/2) prior on sigma2
    float sigma2_scale     = 1.0f;   // lambda
};

struct BARTState {
    int n, p;
    QuantizedX   Xq;    // quantized covariates (owned)
    const float* y;     // outcomes [n]

    std::vector<Tree>               trees;
    std::vector<std::vector<float>> pred;     // pred[t][i]: tree t's prediction for obs i
    std::vector<float>              residual; // partial residual
    float                           sigma2;
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

// Sample leaf values for all leaves of one tree (Gaussian posterior)
void sample_leaves(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, float sigma2, const BARTConfig& cfg, RNG& rng);

} // namespace bart
