#include "faststochtree/sampler.hpp"
#include "faststochtree/mcmc.hpp"

namespace bart {

BARTResult run_bart(const double* X,      const double* y, int n, int p,
                    const double* X_test, int n_test,
                    const BARTConfig& cfg, int n_burnin, int n_samples, RNG& rng) {
    BARTState state;
    state.n = n;
    state.p = p;
    state.X = X;
    state.y = y;

    init_state(state, cfg, rng);

    for (int s = 0; s < n_burnin;  s++) mcmc_sweep(state, cfg, rng);

    BARTResult result;
    result.samples.reserve(n_samples);
    result.sigma2_samples.reserve(n_samples);

    for (int s = 0; s < n_samples; s++) {
        mcmc_sweep(state, cfg, rng);

        // Training predictions: sum cached per-tree predictions
        std::vector<double> pred(n, 0.0);
        for (int t = 0; t < cfg.num_trees; t++)
            for (int i = 0; i < n; i++)
                pred[i] += state.pred[t][i];
        result.samples.push_back(std::move(pred));

        // Test predictions: traverse each test obs through each tree
        std::vector<double> test_pred(n_test, 0.0);
        for (int t = 0; t < cfg.num_trees; t++)
            for (int i = 0; i < n_test; i++)
                test_pred[i] += state.trees[t].nodes[
                    state.trees[t].traverse(X_test, i, p)].value;
        result.test_samples.push_back(std::move(test_pred));

        result.sigma2_samples.push_back(state.sigma2);
        result.forests.push_back(state.trees);  // snapshot: copies T node-vectors
    }

    return result;
}

} // namespace bart
