#pragma once
#include "faststochtree/model.hpp"
#include <vector>

namespace bart {

using Forest = std::vector<Tree>;  // one Forest = T trees at one posterior draw

struct BARTResult {
    std::vector<std::vector<float>> samples;       // [sample_idx][train_obs]
    std::vector<std::vector<float>> test_samples;  // [sample_idx][test_obs]
    std::vector<float>              sigma2_samples; // [sample_idx]
    std::vector<Forest>              forests;        // [sample_idx][tree_idx]
};

BARTResult run_bart(const float* X,      const float* y, int n, int p,
                    const float* X_test, int n_test,
                    const BARTConfig& cfg, int n_burnin, int n_samples, RNG& rng);

} // namespace bart
