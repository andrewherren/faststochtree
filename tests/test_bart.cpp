#include "faststochtree/sampler.hpp"
#include <gtest/gtest.h>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <vector>

// Simulate y = sin(2*pi*x) + N(0, sigma^2), X ~ U(0,1)
static void simulate(int n, int p, unsigned seed,
                     std::vector<float>& X, std::vector<float>& y,
                     float sigma = 1.0f) {
    bart::RNG rng(seed);
    X.resize(n * p);
    y.resize(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) X[i * p + j] = rng.uniform();
        y[i] = std::sin(2.0f * (float)M_PI * X[i * p + 0]) + sigma * rng.normal();
    }
}

TEST(BARTSmoke, RunsWithoutCrash) {
    int n = 200, p = 5;
    std::vector<float> X, y;
    simulate(n, p, 42, X, y);

    bart::BARTConfig cfg;
    cfg.num_trees      = 20;
    cfg.leaf_prior_var = 0.25f / cfg.num_trees;
    cfg.num_threads = 1;

    bart::RNG rng(123);
    auto result = bart::run_bart(X.data(), y.data(), n, p,
                                 /*X_test=*/nullptr, /*n_test=*/0,
                                 cfg, /*n_burnin=*/50, /*n_samples=*/50, rng);

    EXPECT_EQ((int)result.samples.size(),       50);
    EXPECT_EQ((int)result.sigma2_samples.size(), 50);

    // All sigma2 samples should be positive and finite
    for (float s2 : result.sigma2_samples) {
        EXPECT_GT(s2, 0.0f);
        EXPECT_TRUE(std::isfinite(s2));
    }
}

TEST(BARTCorrectness, ReasonableRMSE) {
    int n = 500, p = 5;
    float sigma_true = 1.0f;
    std::vector<float> X, y;
    simulate(n, p, 99, X, y, sigma_true);

    bart::BARTConfig cfg;
    cfg.num_trees      = 50;
    cfg.leaf_prior_var = 0.25f / cfg.num_trees;
    cfg.sigma2_scale   = sigma_true * sigma_true;
    cfg.num_threads    = 1;

    bart::RNG rng(42);
    auto result = bart::run_bart(X.data(), y.data(), n, p,
                                 /*X_test=*/nullptr, /*n_test=*/0,
                                 cfg, /*n_burnin=*/100, /*n_samples=*/100, rng);

    // Posterior mean predictions (accumulate in double to avoid noise)
    std::vector<double> post_mean(n, 0.0);
    int ns = (int)result.samples.size();
    for (auto& sample : result.samples)
        for (int i = 0; i < n; i++)
            post_mean[i] += sample[i] / ns;

    double rss = 0.0;
    for (int i = 0; i < n; i++) {
        double err = y[i] - post_mean[i];
        rss += err * err;
    }
    double rmse = std::sqrt(rss / n);

    // With sigma_true=1 and a signal-to-noise ratio of ~1, RMSE < 2 is a sanity check
    EXPECT_LT(rmse, 2.0) << "RMSE=" << rmse << " is unexpectedly large";

    // Posterior sigma2 should be in a reasonable range
    double sigma2_post = 0.0;
    for (float s2 : result.sigma2_samples) sigma2_post += s2;
    sigma2_post /= ns;
    EXPECT_LT(sigma2_post, 4.0);
    EXPECT_GT(sigma2_post, 0.1);
}
