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

BARTResult run_xbart(const float* X,      const float* y, int n, int p,
                     const float* X_test, int n_test,
                     const BARTConfig& cfg, int n_burnin, int n_samples, RNG& rng);

// ── Public model object for language bindings ─────────────────────────────────
//
// Holds everything needed to make out-of-sample predictions after fitting:
//   - forests:       one Forest (vector<Tree>) per posterior sample
//   - train_cuts:    quantization cut-points from training data, used to map
//                    new observations to the same bin indices at predict time
//   - test_samples:  flat [n_samples × n_test] predictions from the fit call
//   - sigma2_samples: posterior noise-variance draws [n_samples]

struct BARTModel {
    int n_samples = 0;
    int n_test    = 0;

    std::vector<Forest> forests;        // [n_samples][n_trees]
    QuantizedX          train_cuts;     // cut-points only (data field unused)
    std::vector<float>  test_samples;   // flat row-major [n_samples × n_test]
    std::vector<float>  sigma2_samples; // [n_samples]

    // Predict on new data. Returns flat row-major [n_samples × n_new].
    // Each row is one posterior predictive draw; average over rows for E[y|X_new].
    std::vector<float> predict(const float* X_new, int n_new) const;
};

// Convenience constructors — take an integer seed instead of an RNG object.
// Input X is row-major float[n × p]; y is float[n]; X_test is float[n_test × p].
// Pass n_test=0 / X_test=nullptr to skip test predictions at fit time.

BARTModel fit_bart(const float* X,      const float* y, int n, int p,
                   const float* X_test, int n_test,
                   const BARTConfig& cfg, int n_burnin, int n_samples, int seed);

BARTModel fit_xbart(const float* X,      const float* y, int n, int p,
                    const float* X_test, int n_test,
                    const BARTConfig& cfg, int n_burnin, int n_samples, int seed);

} // namespace bart
