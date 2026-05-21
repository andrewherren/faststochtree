#include "faststochtree/sampler.hpp"
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <vector>

// DGP
// ─────────────────────────────────────────────────────────────────────────────
//   mu(x)  = 2 * sin(pi * x1)           uses x1 only
//   tau(x) = 3 * x2                     uses x2 only, range [0,3], sd ~0.87
//   pi(x)  = 0.3 + 0.4 * x1            mild confounding through x1
//   z      ~ Bernoulli(pi(x))
//   y      = mu(x) + tau(x) * z + N(0, sigma^2)
//
// The tau forest has a clear orthogonal signal from the mu forest.
// Null-model CATE RMSE (predict E[tau] = 1.5 everywhere) ≈ 0.87.
// A working BCF on n=600 should reach ~0.3–0.5, well below threshold 0.8.

struct BCFData {
    int n, p;
    std::vector<float> X;        // [n × p] row-major
    std::vector<float> z;        // [n] treatment indicators
    std::vector<float> pi_hat;   // [n] true propensity scores
    std::vector<float> tau_true; // [n] true CATE = 3 * x2
    float y_mean, y_sd;
    std::vector<float> y_scaled; // standardised outcomes passed to BCF
};

static BCFData simulate_bcf(int n, int p, unsigned seed, float sigma = 1.0f) {
    bart::RNG rng(seed);
    BCFData d;
    d.n = n; d.p = p;
    d.X.resize(n * p);
    d.z.resize(n);
    d.pi_hat.resize(n);
    d.tau_true.resize(n);
    std::vector<float> y(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) d.X[i * p + j] = rng.uniform();
        float x1 = d.X[i * p + 0], x2 = d.X[i * p + 1];
        float mu    = 2.0f * std::sin((float)M_PI * x1);
        float tau   = 3.0f * x2;
        float pi    = 0.3f + 0.4f * x1;
        d.pi_hat[i]   = pi;
        d.tau_true[i] = tau;
        d.z[i] = (rng.uniform() < pi) ? 1.0f : 0.0f;
        y[i]   = mu + tau * d.z[i] + sigma * rng.normal();
    }

    // Standardise y so the leaf priors 1/M scale correctly.
    double sum = 0.0, sum2 = 0.0;
    for (float yi : y) { sum += yi; sum2 += (double)yi * yi; }
    d.y_mean = (float)(sum / n);
    d.y_sd   = (float)std::sqrt(sum2 / n - (double)d.y_mean * d.y_mean);
    if (d.y_sd < 1e-6f) d.y_sd = 1.0f;
    d.y_scaled.resize(n);
    for (int i = 0; i < n; i++)
        d.y_scaled[i] = (y[i] - d.y_mean) / d.y_sd;

    return d;
}

// Build X_mu = [X | pi_hat] as [n × (p+1)] row-major.
static std::vector<float> build_X_mu(const std::vector<float>& X, int n, int p,
                                      const std::vector<float>& pi) {
    int p_mu = p + 1;
    std::vector<float> out(n * p_mu);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) out[i * p_mu + j] = X[i * p + j];
        out[i * p_mu + p] = pi[i];
    }
    return out;
}

static bart::BCFConfig default_bcf_cfg() {
    bart::BCFConfig cfg;
    cfg.mu_config.num_trees       = 50;
    cfg.mu_config.leaf_prior_var  = 0.25f / cfg.mu_config.num_trees;
    cfg.tau_config.num_trees      = 20;
    cfg.tau_config.leaf_prior_var = 0.25f / cfg.tau_config.num_trees;
    return cfg;
}

// Compute posterior-mean CATE on the original y scale from result.tau_test.
static std::vector<double> tau_posterior_mean(const bart::BCFResult& result,
                                               int n_test, float y_sd) {
    std::vector<double> out(n_test, 0.0);
    int ns = (int)result.tau_test.size();
    for (const auto& samp : result.tau_test)
        for (int i = 0; i < n_test; i++)
            out[i] += samp[i] / ns;
    for (double& v : out) v *= y_sd;
    return out;
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

TEST(BCFSmoke, MCMCRunsWithoutCrash) {
    BCFData d = simulate_bcf(200, 5, 42);
    auto X_mu = build_X_mu(d.X, d.n, d.p, d.pi_hat);
    auto cfg  = default_bcf_cfg();

    bart::RNG rng(123);
    auto result = bart::run_bcf(X_mu.data(), d.y_scaled.data(), d.z.data(),
                                 d.n, d.p + 1,
                                 d.X.data(), d.p,
                                 nullptr, nullptr, 0,
                                 cfg, 20, 20, rng);

    EXPECT_EQ((int)result.tau_train.size(),      20);
    EXPECT_EQ((int)result.mu_train.size(),       20);
    EXPECT_EQ((int)result.sigma2_samples.size(), 20);
    EXPECT_TRUE(result.tau_test.empty());
    for (float s2 : result.sigma2_samples) {
        EXPECT_GT(s2, 0.0f);
        EXPECT_TRUE(std::isfinite(s2));
    }
    for (const auto& samp : result.tau_train)
        for (float v : samp)
            EXPECT_TRUE(std::isfinite(v));
}

TEST(BCFSmoke, GFRRunsWithoutCrash) {
    BCFData d = simulate_bcf(200, 5, 42);
    auto X_mu = build_X_mu(d.X, d.n, d.p, d.pi_hat);
    auto cfg  = default_bcf_cfg();

    bart::RNG rng(123);
    auto result = bart::run_xbcf(X_mu.data(), d.y_scaled.data(), d.z.data(),
                                  d.n, d.p + 1,
                                  d.X.data(), d.p,
                                  nullptr, nullptr, 0,
                                  cfg, 10, 10, rng);

    EXPECT_EQ((int)result.tau_train.size(),      10);
    EXPECT_EQ((int)result.sigma2_samples.size(), 10);
    for (float s2 : result.sigma2_samples) {
        EXPECT_GT(s2, 0.0f);
        EXPECT_TRUE(std::isfinite(s2));
    }
}

TEST(BCFSmoke, TestSetPredictions) {
    int n = 200, p = 5, n_test = 50;
    BCFData train = simulate_bcf(n,      p, 77);
    BCFData test  = simulate_bcf(n_test, p, 78);
    auto X_mu      = build_X_mu(train.X, n,      p, train.pi_hat);
    auto X_test_mu = build_X_mu(test.X,  n_test, p, test.pi_hat);
    auto cfg = default_bcf_cfg();

    bart::RNG rng(7);
    auto result = bart::run_bcf(X_mu.data(), train.y_scaled.data(), train.z.data(),
                                 n, p + 1,
                                 train.X.data(), p,
                                 X_test_mu.data(), test.X.data(), n_test,
                                 cfg, 20, 20, rng);

    EXPECT_EQ((int)result.tau_test.size(), 20);
    EXPECT_EQ((int)result.mu_test.size(),  20);
    for (const auto& samp : result.tau_test) {
        EXPECT_EQ((int)samp.size(), n_test);
        for (float v : samp) EXPECT_TRUE(std::isfinite(v));
    }
}

// ── Warm-start tests ──────────────────────────────────────────────────────────

TEST(BCFWarmStart, RunsWithoutCrash) {
    BCFData d = simulate_bcf(200, 5, 42);
    auto X_mu = build_X_mu(d.X, d.n, d.p, d.pi_hat);
    auto cfg  = default_bcf_cfg();

    bart::RNG rng(123);
    auto result = bart::run_warmstart_bcf(X_mu.data(), d.y_scaled.data(), d.z.data(),
                                          d.n, d.p + 1,
                                          d.X.data(), d.p,
                                          nullptr, nullptr, 0,
                                          cfg, 10, 0, 20, rng);

    EXPECT_EQ((int)result.tau_train.size(),      20);
    EXPECT_EQ((int)result.sigma2_samples.size(), 20);
    for (float s2 : result.sigma2_samples) {
        EXPECT_GT(s2, 0.0f);
        EXPECT_TRUE(std::isfinite(s2));
    }
}

TEST(BCFWarmStart, KeepGFRSamples) {
    BCFData d = simulate_bcf(200, 5, 42);
    auto X_mu = build_X_mu(d.X, d.n, d.p, d.pi_hat);
    auto cfg  = default_bcf_cfg();

    bart::RNG rng(7);
    auto result = bart::run_warmstart_bcf(X_mu.data(), d.y_scaled.data(), d.z.data(),
                                          d.n, d.p + 1,
                                          d.X.data(), d.p,
                                          nullptr, nullptr, 0,
                                          cfg, 5, 0, 20, rng,
                                          /*keep_gfr_samples=*/true);

    // 5 GFR draws prepended + 20 MCMC draws = 25 total
    EXPECT_EQ((int)result.tau_train.size(),      25);
    EXPECT_EQ((int)result.sigma2_samples.size(), 25);
}

// ── CATE recovery tests ───────────────────────────────────────────────────────

// Shared correctness check: posterior-mean CATE RMSE < 0.8 on held-out data.
// The null-model RMSE (predicting E[tau]=1.5 everywhere) is ~0.87, so this
// threshold requires meaningful recovery.
static void check_cate_recovery(const bart::BCFResult& result,
                                  const BCFData& test, float y_sd) {
    int n_test = test.n;
    auto tau_hat = tau_posterior_mean(result, n_test, y_sd);

    double rss = 0.0;
    for (int i = 0; i < n_test; i++) {
        double err = tau_hat[i] - (double)test.tau_true[i];
        rss += err * err;
    }
    double rmse = std::sqrt(rss / n_test);
    EXPECT_LT(rmse, 0.8) << "CATE RMSE=" << rmse << " (null-model baseline ~0.87)";

    // Average CATE should be near E[3*x2] = 1.5
    double mean = 0.0;
    for (double v : tau_hat) mean += v;
    mean /= n_test;
    EXPECT_GT(mean, 0.5) << "Average CATE should be ~1.5";
    EXPECT_LT(mean, 2.5) << "Average CATE should be ~1.5";
}

TEST(BCFCorrectness, MCMCCATERecovery) {
    int n = 600, p = 5, n_test = 200;
    BCFData train = simulate_bcf(n,      p, 10);
    BCFData test  = simulate_bcf(n_test, p, 11);
    auto X_mu      = build_X_mu(train.X, n,      p, train.pi_hat);
    auto X_test_mu = build_X_mu(test.X,  n_test, p, test.pi_hat);
    auto cfg = default_bcf_cfg();

    bart::RNG rng(42);
    auto result = bart::run_bcf(X_mu.data(), train.y_scaled.data(), train.z.data(),
                                 n, p + 1,
                                 train.X.data(), p,
                                 X_test_mu.data(), test.X.data(), n_test,
                                 cfg, 100, 200, rng);

    EXPECT_EQ((int)result.tau_test.size(), 200);
    check_cate_recovery(result, test, train.y_sd);
}

TEST(BCFCorrectness, WarmStartCATERecovery) {
    int n = 600, p = 5, n_test = 200;
    BCFData train = simulate_bcf(n,      p, 20);
    BCFData test  = simulate_bcf(n_test, p, 21);
    auto X_mu      = build_X_mu(train.X, n,      p, train.pi_hat);
    auto X_test_mu = build_X_mu(test.X,  n_test, p, test.pi_hat);
    auto cfg = default_bcf_cfg();

    bart::RNG rng(99);
    auto result = bart::run_warmstart_bcf(X_mu.data(), train.y_scaled.data(), train.z.data(),
                                          n, p + 1,
                                          train.X.data(), p,
                                          X_test_mu.data(), test.X.data(), n_test,
                                          cfg, 20, 0, 200, rng);

    EXPECT_EQ((int)result.tau_test.size(), 200);
    check_cate_recovery(result, test, train.y_sd);
}
