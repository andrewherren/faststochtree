#include "faststochtree/sampler.hpp"
#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/thread_pool.hpp"
#include <memory>

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
                    test_pred[i] += state.trees[t].leaf_value[
                        state.trees[t].traverse(test_qx.data.data(), i, n_test)];
            result.test_samples.push_back(std::move(test_pred));
        }

        result.sigma2_samples.push_back(state.sigma2);
        result.forests.push_back(state.trees);
    }

    return result;
}

BARTResult run_xbart(const float* X,      const float* y, int n, int p,
                     const float* X_test, int n_test,
                     const BARTConfig& cfg, int n_burnin, int n_samples, RNG& rng) {
    BARTState state;
    state.n  = n;
    state.p  = p;
    state.Xq = quantize(X, n, p);
    state.y  = y;

    init_state(state, cfg, rng);
    state.gfr_hist_ws.alloc(n, p, state.trees[0].full_size);  // allocate persistent histogram workspace
    if (cfg.num_threads > 1)
        state.thread_pool = std::make_unique<ThreadPool>(cfg.num_threads);

    QuantizedX test_qx;
    if (n_test > 0 && X_test != nullptr)
        test_qx = quantize_with_cuts(X_test, n_test, state.Xq);

    for (int s = 0; s < n_burnin; s++) gfr_sweep(state, cfg, rng);

    BARTResult result;
    result.samples.reserve(n_samples);
    result.sigma2_samples.reserve(n_samples);

    for (int s = 0; s < n_samples; s++) {
        gfr_sweep(state, cfg, rng);

        std::vector<float> pred(n, 0.f);
        for (int t = 0; t < cfg.num_trees; t++)
            for (int i = 0; i < n; i++)
                pred[i] += state.pred[t][i];
        result.samples.push_back(std::move(pred));

        if (n_test > 0) {
            std::vector<float> test_pred(n_test, 0.f);
            for (int t = 0; t < cfg.num_trees; t++)
                for (int i = 0; i < n_test; i++)
                    test_pred[i] += state.trees[t].leaf_value[
                        state.trees[t].traverse(test_qx.data.data(), i, n_test)];
            result.test_samples.push_back(std::move(test_pred));
        }

        result.sigma2_samples.push_back(state.sigma2);
        result.forests.push_back(state.trees);
    }

    return result;
}


// ── BARTModel::predict ────────────────────────────────────────────────────────

std::vector<float> BARTModel::predict(const float* X_new, int n_new) const {
    QuantizedX qx = quantize_with_cuts(X_new, n_new, train_cuts);
    std::vector<float> out(n_samples * n_new, 0.f);
    for (int s = 0; s < n_samples; s++) {
        const Forest& forest = forests[s];
        float* row = out.data() + s * n_new;
        for (const Tree& tree : forest)
            for (int i = 0; i < n_new; i++)
                row[i] += tree.leaf_value[tree.traverse(qx.data.data(), i, n_new)];
    }
    return out;
}

// ── fit_bart / fit_xbart ──────────────────────────────────────────────────────

static BARTModel make_model(BARTResult& res, QuantizedX train_cuts) {
    BARTModel m;
    m.n_samples      = static_cast<int>(res.forests.size());
    m.n_test         = res.test_samples.empty() ? 0
                       : static_cast<int>(res.test_samples[0].size());
    m.forests        = std::move(res.forests);
    m.train_cuts     = std::move(train_cuts);
    m.sigma2_samples = std::move(res.sigma2_samples);

    // Flatten test_samples: [sample][obs] → row-major [n_samples × n_test]
    m.test_samples.resize(static_cast<size_t>(m.n_samples) * m.n_test);
    for (int s = 0; s < m.n_samples; s++)
        std::copy(res.test_samples[s].begin(), res.test_samples[s].end(),
                  m.test_samples.data() + s * m.n_test);
    return m;
}

BARTModel fit_bart(const float* X,      const float* y, int n, int p,
                   const float* X_test, int n_test,
                   const BARTConfig& cfg, int n_burnin, int n_samples, int seed) {
    RNG rng(static_cast<unsigned>(seed));
    // Quantize training data first so we can save the cuts.
    QuantizedX train_qx = quantize(X, n, p);
    // run_bart re-quantizes internally, but we need the cuts separately.
    // Re-use the same quantized data by calling the internal path directly.
    BARTResult res = run_bart(X, y, n, p, X_test, n_test, cfg, n_burnin, n_samples, rng);
    // Extract cuts from a fresh quantize call (same data → same cuts).
    QuantizedX cuts_only = std::move(train_qx);
    cuts_only.data.clear();  // drop raw data, keep cuts
    return make_model(res, std::move(cuts_only));
}

BARTModel fit_xbart(const float* X,      const float* y, int n, int p,
                    const float* X_test, int n_test,
                    const BARTConfig& cfg, int n_burnin, int n_samples, int seed) {
    RNG rng(static_cast<unsigned>(seed));
    QuantizedX train_qx = quantize(X, n, p);
    BARTResult res = run_xbart(X, y, n, p, X_test, n_test, cfg, n_burnin, n_samples, rng);
    QuantizedX cuts_only = std::move(train_qx);
    cuts_only.data.clear();
    return make_model(res, std::move(cuts_only));
}

} // namespace bart
