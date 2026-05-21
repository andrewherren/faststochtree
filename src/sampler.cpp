#include "faststochtree/sampler.hpp"
#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/thread_pool.hpp"
#include <algorithm>
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


BARTResult run_warmstart_bart(const float* X,      const float* y, int n, int p,
                               const float* X_test, int n_test,
                               const BARTConfig& cfg,
                               int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                               RNG& rng, bool keep_gfr_samples) {
    BARTState state;
    state.n  = n;
    state.p  = p;
    state.Xq = quantize(X, n, p);
    state.y  = y;

    init_state(state, cfg, rng);
    state.gfr_hist_ws.alloc(n, p, state.trees[0].full_size);
    if (cfg.num_threads > 1)
        state.thread_pool = std::make_unique<ThreadPool>(cfg.num_threads);

    QuantizedX test_qx;
    if (n_test > 0 && X_test != nullptr)
        test_qx = quantize_with_cuts(X_test, n_test, state.Xq);

    BARTResult result;
    result.samples.reserve(n_samples + (keep_gfr_samples ? n_gfr_burnin : 0));
    result.sigma2_samples.reserve(n_samples + (keep_gfr_samples ? n_gfr_burnin : 0));

    auto collect = [&]() {
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
    };

    for (int s = 0; s < n_gfr_burnin; s++) {
        gfr_sweep(state, cfg, rng);
        if (keep_gfr_samples) collect();
    }

    for (int s = 0; s < n_mcmc_burnin; s++) mcmc_sweep(state, cfg, rng);

    for (int s = 0; s < n_samples; s++) {
        mcmc_sweep(state, cfg, rng);
        collect();
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

BARTModel fit_warmstart_bart(const float* X,      const float* y, int n, int p,
                              const float* X_test, int n_test,
                              const BARTConfig& cfg,
                              int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                              int seed, bool keep_gfr_samples) {
    RNG rng(static_cast<unsigned>(seed));
    QuantizedX train_qx = quantize(X, n, p);
    BARTResult res = run_warmstart_bart(X, y, n, p, X_test, n_test, cfg,
                                        n_gfr_burnin, n_mcmc_burnin, n_samples,
                                        rng, keep_gfr_samples);
    QuantizedX cuts_only = std::move(train_qx);
    cuts_only.data.clear();
    return make_model(res, std::move(cuts_only));
}

// ── BCF ──────────────────────────────────────────────────────────────────────

// Build a row-major [n × (p+1)] matrix by appending pi_hat as the last column.
static std::vector<float> append_column(const float* X, int n, int p,
                                         const float* pi_hat) {
    std::vector<float> out(n * (p + 1));
    for (int i = 0; i < n; i++) {
        std::copy(X + i * p, X + i * p + p, out.data() + i * (p + 1));
        out[i * (p + 1) + p] = pi_hat[i];
    }
    return out;
}

static void init_bcf_states(BARTState& mu_s, const BARTConfig& mu_cfg,
                              BARTState& tau_s, const BARTConfig& tau_cfg,
                              const float* y, int n,
                              std::vector<float>& shared_resid, float& sigma2,
                              RNG& rng, bool alloc_gfr) {
    init_state(mu_s, mu_cfg, rng);
    init_state(tau_s, tau_cfg, rng);
    if (alloc_gfr) {
        mu_s.gfr_hist_ws.alloc(n, mu_s.p, mu_s.trees[0].full_size);
        tau_s.gfr_hist_ws.alloc(n, tau_s.p, tau_s.trees[0].full_size, true);
        if (mu_cfg.num_threads > 1)
            mu_s.thread_pool = std::make_unique<ThreadPool>(mu_cfg.num_threads);
        if (tau_cfg.num_threads > 1)
            tau_s.thread_pool = std::make_unique<ThreadPool>(tau_cfg.num_threads);
    }
    shared_resid.assign(y, y + n);
    sigma2 = 1.f;
}

static void collect_bcf(const BARTState& mu_s, const BARTState& tau_s,
                          const QuantizedX& test_mu_qx, const QuantizedX& test_tau_qx,
                          int n, int n_test, float sigma2,
                          BCFResult& result) {
    std::vector<float> mu_tr(n, 0.f), tau_tr(n, 0.f);
    for (int t = 0; t < (int)mu_s.pred.size();  t++)
        for (int i = 0; i < n; i++) mu_tr[i] += mu_s.pred[t][i];
    // tau_s.pred[t][i] = β_leaf·z_i; collect raw β_leaf for CATE
    for (int t = 0; t < (int)tau_s.trees.size(); t++)
        for (int i = 0; i < n; i++)
            tau_tr[i] += tau_s.trees[t].leaf_value[tau_s.leaf_indices[t][i]];
    result.mu_train.push_back(std::move(mu_tr));
    result.tau_train.push_back(std::move(tau_tr));

    if (n_test > 0) {
        std::vector<float> mu_te(n_test, 0.f), tau_te(n_test, 0.f);
        for (const Tree& tr : mu_s.trees)
            for (int i = 0; i < n_test; i++)
                mu_te[i] += tr.leaf_value[tr.traverse(test_mu_qx.data.data(), i, n_test)];
        for (const Tree& tr : tau_s.trees)
            for (int i = 0; i < n_test; i++)
                tau_te[i] += tr.leaf_value[tr.traverse(test_tau_qx.data.data(), i, n_test)];
        result.mu_test.push_back(std::move(mu_te));
        result.tau_test.push_back(std::move(tau_te));
    }

    result.sigma2_samples.push_back(sigma2);
    result.mu_forests.push_back(mu_s.trees);
    result.tau_forests.push_back(tau_s.trees);
}

BCFResult run_bcf(const float* X_mu,  const float* y, const float* z,
                  int n, int p_mu,
                  const float* X_tau, int p_tau,
                  const float* X_test_mu,  const float* X_test_tau, int n_test,
                  const BCFConfig& cfg, int n_burnin, int n_samples, RNG& rng) {
    BARTState mu_s, tau_s;
    mu_s.n = tau_s.n = n;
    mu_s.p = p_mu; tau_s.p = p_tau;
    mu_s.Xq = quantize(X_mu, n, p_mu);
    tau_s.Xq = quantize(X_tau, n, p_tau);
    mu_s.y = tau_s.y = y;

    std::vector<float> shared_resid; float sigma2;
    init_bcf_states(mu_s, cfg.mu_config, tau_s, cfg.tau_config,
                    y, n, shared_resid, sigma2, rng, false);

    QuantizedX test_mu_qx, test_tau_qx;
    if (n_test > 0 && X_test_mu && X_test_tau) {
        test_mu_qx  = quantize_with_cuts(X_test_mu,  n_test, mu_s.Xq);
        test_tau_qx = quantize_with_cuts(X_test_tau, n_test, tau_s.Xq);
    }

    for (int s = 0; s < n_burnin; s++)
        bcf_mcmc_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);

    BCFResult result;
    for (int s = 0; s < n_samples; s++) {
        bcf_mcmc_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);
        collect_bcf(mu_s, tau_s, test_mu_qx, test_tau_qx, n, n_test, sigma2, result);
    }
    return result;
}

BCFResult run_xbcf(const float* X_mu,  const float* y, const float* z,
                   int n, int p_mu,
                   const float* X_tau, int p_tau,
                   const float* X_test_mu,  const float* X_test_tau, int n_test,
                   const BCFConfig& cfg, int n_burnin, int n_samples, RNG& rng) {
    BARTState mu_s, tau_s;
    mu_s.n = tau_s.n = n;
    mu_s.p = p_mu; tau_s.p = p_tau;
    mu_s.Xq = quantize(X_mu, n, p_mu);
    tau_s.Xq = quantize(X_tau, n, p_tau);
    mu_s.y = tau_s.y = y;

    std::vector<float> shared_resid; float sigma2;
    init_bcf_states(mu_s, cfg.mu_config, tau_s, cfg.tau_config,
                    y, n, shared_resid, sigma2, rng, true);

    QuantizedX test_mu_qx, test_tau_qx;
    if (n_test > 0 && X_test_mu && X_test_tau) {
        test_mu_qx  = quantize_with_cuts(X_test_mu,  n_test, mu_s.Xq);
        test_tau_qx = quantize_with_cuts(X_test_tau, n_test, tau_s.Xq);
    }

    for (int s = 0; s < n_burnin; s++)
        bcf_gfr_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);

    BCFResult result;
    for (int s = 0; s < n_samples; s++) {
        bcf_gfr_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);
        collect_bcf(mu_s, tau_s, test_mu_qx, test_tau_qx, n, n_test, sigma2, result);
    }
    return result;
}

BCFResult run_warmstart_bcf(const float* X_mu,  const float* y, const float* z,
                             int n, int p_mu,
                             const float* X_tau, int p_tau,
                             const float* X_test_mu,  const float* X_test_tau, int n_test,
                             const BCFConfig& cfg,
                             int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                             RNG& rng, bool keep_gfr_samples) {
    BARTState mu_s, tau_s;
    mu_s.n = tau_s.n = n;
    mu_s.p = p_mu; tau_s.p = p_tau;
    mu_s.Xq = quantize(X_mu, n, p_mu);
    tau_s.Xq = quantize(X_tau, n, p_tau);
    mu_s.y = tau_s.y = y;

    std::vector<float> shared_resid; float sigma2;
    init_bcf_states(mu_s, cfg.mu_config, tau_s, cfg.tau_config,
                    y, n, shared_resid, sigma2, rng, true);

    QuantizedX test_mu_qx, test_tau_qx;
    if (n_test > 0 && X_test_mu && X_test_tau) {
        test_mu_qx  = quantize_with_cuts(X_test_mu,  n_test, mu_s.Xq);
        test_tau_qx = quantize_with_cuts(X_test_tau, n_test, tau_s.Xq);
    }

    BCFResult result;
    for (int s = 0; s < n_gfr_burnin; s++) {
        bcf_gfr_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);
        if (keep_gfr_samples)
            collect_bcf(mu_s, tau_s, test_mu_qx, test_tau_qx, n, n_test, sigma2, result);
    }
    for (int s = 0; s < n_mcmc_burnin; s++)
        bcf_mcmc_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);
    for (int s = 0; s < n_samples; s++) {
        bcf_mcmc_sweep(shared_resid, z, mu_s, cfg.mu_config, tau_s, cfg.tau_config, sigma2, rng);
        collect_bcf(mu_s, tau_s, test_mu_qx, test_tau_qx, n, n_test, sigma2, result);
    }
    return result;
}

// ── BCFModel::predict_tau / predict_mu ───────────────────────────────────────

std::vector<float> BCFModel::predict_tau(const float* X_new, int n_new) const {
    QuantizedX qx = quantize_with_cuts(X_new, n_new, tau_train_cuts);
    std::vector<float> out(n_samples * n_new, 0.f);
    for (int s = 0; s < n_samples; s++) {
        float* row = out.data() + s * n_new;
        for (const Tree& tree : tau_forests[s])
            for (int i = 0; i < n_new; i++)
                row[i] += tree.leaf_value[tree.traverse(qx.data.data(), i, n_new)];
    }
    return out;
}

std::vector<float> BCFModel::predict_mu(const float* X_new_mu, int n_new) const {
    QuantizedX qx = quantize_with_cuts(X_new_mu, n_new, mu_train_cuts);
    std::vector<float> out(n_samples * n_new, 0.f);
    for (int s = 0; s < n_samples; s++) {
        float* row = out.data() + s * n_new;
        for (const Tree& tree : mu_forests[s])
            for (int i = 0; i < n_new; i++)
                row[i] += tree.leaf_value[tree.traverse(qx.data.data(), i, n_new)];
    }
    return out;
}

// ── fit_bcf / fit_xbcf / fit_warmstart_bcf ───────────────────────────────────

static BCFModel make_bcf_model(BCFResult& res,
                                QuantizedX mu_cuts, QuantizedX tau_cuts) {
    BCFModel m;
    m.n_samples      = static_cast<int>(res.mu_forests.size());
    m.n_test         = res.mu_test.empty() ? 0
                       : static_cast<int>(res.mu_test[0].size());
    m.mu_forests     = std::move(res.mu_forests);
    m.tau_forests    = std::move(res.tau_forests);
    m.mu_train_cuts  = std::move(mu_cuts);
    m.tau_train_cuts = std::move(tau_cuts);
    m.sigma2_samples = std::move(res.sigma2_samples);

    int ns = m.n_samples, nt = m.n_test;
    m.test_mu.resize(static_cast<size_t>(ns) * nt);
    m.test_tau.resize(static_cast<size_t>(ns) * nt);
    for (int s = 0; s < ns; s++) {
        if (nt > 0) {
            std::copy(res.mu_test[s].begin(),  res.mu_test[s].end(),
                      m.test_mu.data()  + s * nt);
            std::copy(res.tau_test[s].begin(), res.tau_test[s].end(),
                      m.test_tau.data() + s * nt);
        }
    }
    return m;
}

BCFModel fit_bcf(const float* X,      const float* y, const float* z,
                 const float* pi_hat, int n, int p,
                 const float* X_test, const float* pi_hat_test, int n_test,
                 const BCFConfig& cfg, int n_burnin, int n_samples, int seed) {
    RNG rng(static_cast<unsigned>(seed));
    std::vector<float> X_mu_v = append_column(X, n, p, pi_hat);
    QuantizedX mu_cuts = quantize(X_mu_v.data(), n, p + 1);
    mu_cuts.data.clear();
    QuantizedX tau_cuts = quantize(X, n, p);
    tau_cuts.data.clear();

    std::vector<float> X_test_mu_v;
    const float* X_test_mu  = nullptr;
    const float* X_test_tau = nullptr;
    if (n_test > 0 && X_test && pi_hat_test) {
        X_test_mu_v = append_column(X_test, n_test, p, pi_hat_test);
        X_test_mu   = X_test_mu_v.data();
        X_test_tau  = X_test;
    }

    BCFResult res = run_bcf(X_mu_v.data(), y, z, n, p + 1,
                             X, p,
                             X_test_mu, X_test_tau, n_test,
                             cfg, n_burnin, n_samples, rng);
    return make_bcf_model(res, std::move(mu_cuts), std::move(tau_cuts));
}

BCFModel fit_xbcf(const float* X,      const float* y, const float* z,
                  const float* pi_hat, int n, int p,
                  const float* X_test, const float* pi_hat_test, int n_test,
                  const BCFConfig& cfg, int n_burnin, int n_samples, int seed) {
    RNG rng(static_cast<unsigned>(seed));
    std::vector<float> X_mu_v = append_column(X, n, p, pi_hat);
    QuantizedX mu_cuts = quantize(X_mu_v.data(), n, p + 1);
    mu_cuts.data.clear();
    QuantizedX tau_cuts = quantize(X, n, p);
    tau_cuts.data.clear();

    std::vector<float> X_test_mu_v;
    const float* X_test_mu  = nullptr;
    const float* X_test_tau = nullptr;
    if (n_test > 0 && X_test && pi_hat_test) {
        X_test_mu_v = append_column(X_test, n_test, p, pi_hat_test);
        X_test_mu   = X_test_mu_v.data();
        X_test_tau  = X_test;
    }

    BCFResult res = run_xbcf(X_mu_v.data(), y, z, n, p + 1,
                              X, p,
                              X_test_mu, X_test_tau, n_test,
                              cfg, n_burnin, n_samples, rng);
    return make_bcf_model(res, std::move(mu_cuts), std::move(tau_cuts));
}

BCFModel fit_warmstart_bcf(const float* X,      const float* y, const float* z,
                            const float* pi_hat, int n, int p,
                            const float* X_test, const float* pi_hat_test, int n_test,
                            const BCFConfig& cfg,
                            int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                            int seed, bool keep_gfr_samples) {
    RNG rng(static_cast<unsigned>(seed));
    std::vector<float> X_mu_v = append_column(X, n, p, pi_hat);
    QuantizedX mu_cuts = quantize(X_mu_v.data(), n, p + 1);
    mu_cuts.data.clear();
    QuantizedX tau_cuts = quantize(X, n, p);
    tau_cuts.data.clear();

    std::vector<float> X_test_mu_v;
    const float* X_test_mu  = nullptr;
    const float* X_test_tau = nullptr;
    if (n_test > 0 && X_test && pi_hat_test) {
        X_test_mu_v = append_column(X_test, n_test, p, pi_hat_test);
        X_test_mu   = X_test_mu_v.data();
        X_test_tau  = X_test;
    }

    BCFResult res = run_warmstart_bcf(X_mu_v.data(), y, z, n, p + 1,
                                       X, p,
                                       X_test_mu, X_test_tau, n_test,
                                       cfg, n_gfr_burnin, n_mcmc_burnin, n_samples,
                                       rng, keep_gfr_samples);
    return make_bcf_model(res, std::move(mu_cuts), std::move(tau_cuts));
}

} // namespace bart
