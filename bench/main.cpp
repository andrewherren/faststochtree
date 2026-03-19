#include "faststochtree/sampler.hpp"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static const char* get_arg(int argc, char** argv, const char* key, const char* def) {
    int klen = (int)strlen(key);
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--", 2) != 0) continue;
        const char* arg = argv[i] + 2;
        if (strncmp(arg, key, klen) == 0) {
            if (arg[klen] == '=') return arg + klen + 1;
            if (arg[klen] == '\0' && i + 1 < argc) return argv[i + 1];
        }
    }
    return def;
}

int main(int argc, char** argv) {
    int   n_train    = atoi(get_arg(argc, argv, "n_train",  "50000"));
    int   n_test     = atoi(get_arg(argc, argv, "n_test",   "500"));
    int   p          = atoi(get_arg(argc, argv, "p",        "50"));
    int   num_trees  = atoi(get_arg(argc, argv, "trees",    "200"));
    int   n_burnin   = atoi(get_arg(argc, argv, "burnin",   "200"));
    int   n_samples  = atoi(get_arg(argc, argv, "samples",  "1000"));
    int   seed       = atoi(get_arg(argc, argv, "seed",     "12345"));
    float sigma_true = atof(get_arg(argc, argv, "sigma",    "1.0"));

    int n_total = n_train + n_test;

    // Simulate data: y = sin(2*pi*x1) + N(0, sigma_true^2)
    bart::RNG data_rng(seed);
    std::vector<float> X(n_total * p), y_all(n_total);
    for (int i = 0; i < n_total; i++) {
        for (int j = 0; j < p; j++) X[i * p + j] = data_rng.uniform();
        float signal = std::sin(2.f * (float)M_PI * X[i * p + 0]);
        y_all[i] = signal + sigma_true * data_rng.normal();
    }

    // tau: leaf prior variance — calibrated so ensemble prior std ≈ 0.5
    float tau = (0.5f * 0.5f) / num_trees;

    bart::BARTConfig cfg;
    cfg.num_trees      = num_trees;
    cfg.leaf_prior_var = tau;
    cfg.sigma2_shape   = 3.0f;
    cfg.sigma2_scale   = sigma_true * sigma_true;

    bart::RNG model_rng(seed + 1);

    printf("faststochtree BART benchmark (v5-fixed-depth)\n");
    printf("  n_train=%d  n_test=%d  p=%d  trees=%d\n",
           n_train, n_test, p, num_trees);
    printf("  burnin=%d  samples=%d  sigma_true=%.2f\n",
           n_burnin, n_samples, sigma_true);

    auto t0 = std::chrono::steady_clock::now();
    printf("Running %d burnin iterations + %d samples...", n_burnin, n_samples);
    fflush(stdout);

    const float* X_test = X.data() + n_train * p;
    auto result = bart::run_bart(X.data(), y_all.data(), n_train, p,
                                 X_test, n_test,
                                 cfg, n_burnin, n_samples, model_rng);

    auto t1 = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf(" done in %ld ms\n", ms);

    // Posterior mean over test samples — accumulate in double to avoid noise
    std::vector<double> test_post_mean(n_test, 0.0);
    for (auto& sample : result.test_samples)
        for (int i = 0; i < n_test; i++)
            test_post_mean[i] += sample[i] / n_samples;

    double rss = 0.0;
    const float* y_test = y_all.data() + n_train;
    for (int i = 0; i < n_test; i++) {
        double err = y_test[i] - test_post_mean[i];
        rss += err * err;
    }
    double rmse = std::sqrt(rss / n_test);

    double sigma2_post = 0.0;
    for (float s2 : result.sigma2_samples) sigma2_post += s2;
    sigma2_post /= n_samples;

    printf("\nTest RMSE:       %.4f  (sigma_true=%.4f)\n", rmse, sigma_true);
    printf("Posterior sigma: %.4f\n", std::sqrt(sigma2_post));
    printf("Total time:      %ld ms\n", ms);

    return 0;
}
