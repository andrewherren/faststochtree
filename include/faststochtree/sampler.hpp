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

// GFR burn-in followed by MCMC sampling. Runs n_gfr_burnin GFR sweeps to
// warm-start the forest state, then n_mcmc_burnin additional MCMC sweeps
// (typically 0), then collects n_samples MCMC draws. If keep_gfr_samples is
// true, the GFR burn-in draws are prepended to the result forests/samples.
BARTResult run_warmstart_bart(const float* X,      const float* y, int n, int p,
                               const float* X_test, int n_test,
                               const BARTConfig& cfg,
                               int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                               RNG& rng, bool keep_gfr_samples = false);

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

BARTModel fit_warmstart_bart(const float* X,      const float* y, int n, int p,
                              const float* X_test, int n_test,
                              const BARTConfig& cfg,
                              int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                              int seed, bool keep_gfr_samples = false);

// ── BCF ──────────────────────────────────────────────────────────────────────

// Intermediate result type for run_bcf / run_xbcf / run_warmstart_bcf.
struct BCFResult {
    std::vector<std::vector<float>> mu_train;   // [sample][n_train]
    std::vector<std::vector<float>> tau_train;  // [sample][n_train] — β_leaf, NOT z-weighted
    std::vector<std::vector<float>> mu_test;    // [sample][n_test]
    std::vector<std::vector<float>> tau_test;   // [sample][n_test] — β_leaf (CATE)
    std::vector<float>              sigma2_samples;
    std::vector<Forest>             mu_forests;
    std::vector<Forest>             tau_forests;
};

// Public BCF model object returned by fit_bcf / fit_xbcf / fit_warmstart_bcf.
// mu forest covariates are X augmented with pi_hat (p+1 features).
// tau forest covariates are X only (p features).
// tau predictions are CATE estimates β_leaf(x) — not multiplied by z.
struct BCFModel {
    int n_samples = 0;
    int n_test    = 0;

    std::vector<Forest> mu_forests;
    std::vector<Forest> tau_forests;
    QuantizedX          mu_train_cuts;   // cut-points for μ forest (p+1 features)
    QuantizedX          tau_train_cuts;  // cut-points for τ forest (p features)
    std::vector<float>  test_mu;         // flat [n_samples × n_test]
    std::vector<float>  test_tau;        // flat [n_samples × n_test] — CATE
    std::vector<float>  sigma2_samples;  // [n_samples]

    // Predict CATE τ(x_new) on new observations. Returns [n_samples × n_new].
    std::vector<float> predict_tau(const float* X_new, int n_new) const;
    // Predict μ(x_new, pi_hat_new). X_new_mu must be [n_new × (p+1)].
    std::vector<float> predict_mu(const float* X_new_mu, int n_new) const;
};

// Internal run functions (accept an RNG object; used by fit_* wrappers below).
BCFResult run_bcf(const float* X_mu,  const float* y, const float* z,
                  int n, int p_mu,
                  const float* X_tau, int p_tau,
                  const float* X_test_mu,  const float* X_test_tau, int n_test,
                  const BCFConfig& cfg, int n_burnin, int n_samples, RNG& rng);

BCFResult run_xbcf(const float* X_mu,  const float* y, const float* z,
                   int n, int p_mu,
                   const float* X_tau, int p_tau,
                   const float* X_test_mu,  const float* X_test_tau, int n_test,
                   const BCFConfig& cfg, int n_burnin, int n_samples, RNG& rng);

BCFResult run_warmstart_bcf(const float* X_mu,  const float* y, const float* z,
                             int n, int p_mu,
                             const float* X_tau, int p_tau,
                             const float* X_test_mu,  const float* X_test_tau, int n_test,
                             const BCFConfig& cfg,
                             int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                             RNG& rng, bool keep_gfr_samples = false);

// Convenience wrappers: accept X (n×p), y, z, pi_hat (n), X_test (n_test×p),
// pi_hat_test (n_test) and build X_mu = [X | pi_hat] internally.
// Pass n_test=0 / X_test=nullptr / pi_hat_test=nullptr to skip test predictions.

BCFModel fit_bcf(const float* X,         const float* y, const float* z,
                 const float* pi_hat,    int n, int p,
                 const float* X_test,    const float* pi_hat_test, int n_test,
                 const BCFConfig& cfg,   int n_burnin, int n_samples, int seed);

BCFModel fit_xbcf(const float* X,        const float* y, const float* z,
                  const float* pi_hat,   int n, int p,
                  const float* X_test,   const float* pi_hat_test, int n_test,
                  const BCFConfig& cfg,  int n_burnin, int n_samples, int seed);

BCFModel fit_warmstart_bcf(const float* X,       const float* y, const float* z,
                            const float* pi_hat,  int n, int p,
                            const float* X_test,  const float* pi_hat_test, int n_test,
                            const BCFConfig& cfg,
                            int n_gfr_burnin, int n_mcmc_burnin, int n_samples,
                            int seed, bool keep_gfr_samples = false);

} // namespace bart
