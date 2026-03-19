#include "faststochtree/sampler.hpp"
#include "faststochtree/mcmc.hpp"
#include "faststochtree/quantize.hpp"

namespace bart {

BARTResult run_bart(const float* X,      const float* y, int n, int p,
                    const float* X_test, int n_test,
                    const BARTConfig& cfg, int n_burnin, int n_samples, RNG& rng) {
    BARTState state;
    state.n  = n;
    state.p  = p;
    state.Xq = quantize(X, n, p);
    state.y  = y;

    init_state(state, cfg, rng);

    // Quantize test data using the training cut-point tables
    QuantizedX test_qx;
    if (n_test > 0 && X_test != nullptr)
        test_qx = quantize_with_cuts(X_test, n_test, state.Xq);

    for (int s = 0; s < n_burnin; s++) mcmc_sweep(state, cfg, rng);

    BARTResult result;
    result.samples.reserve(n_samples);
    result.sigma2_samples.reserve(n_samples);

    for (int s = 0; s < n_samples; s++) {
        mcmc_sweep(state, cfg, rng);

        // Training predictions: sum cached per-tree predictions
        std::vector<float> pred(n, 0.f);
        for (int t = 0; t < cfg.num_trees; t++)
            for (int i = 0; i < n; i++)
                pred[i] += state.pred[t][i];  // cached; no traversal needed
        result.samples.push_back(std::move(pred));

        // Test predictions: traverse each test obs through each tree
        if (n_test > 0) {
            std::vector<float> test_pred(n_test, 0.f);
            for (int t = 0; t < cfg.num_trees; t++)
                for (int i = 0; i < n_test; i++)
                    test_pred[i] += state.trees[t].nodes[
                        state.trees[t].traverse(test_qx.data.data(), i, n_test)].value;
            result.test_samples.push_back(std::move(test_pred));
        }

        result.sigma2_samples.push_back(state.sigma2);
        result.forests.push_back(state.trees);
    }

    return result;
}

} // namespace bart
