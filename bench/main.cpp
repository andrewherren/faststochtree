#include "faststochtree/sampler.hpp"
#include <chrono>
#include <cmath>
#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
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
    const char* mode     = get_arg(argc, argv, "mode",    "bart");
    int   n_train        = atoi(get_arg(argc, argv, "n_train",  "50000"));
    int   n_test         = atoi(get_arg(argc, argv, "n_test",   "500"));
    int   p              = atoi(get_arg(argc, argv, "p",        "50"));
    int   num_trees      = atoi(get_arg(argc, argv, "trees",    "200"));
    int   seed           = atoi(get_arg(argc, argv, "seed",     "12345"));
    int   num_threads    = atoi(get_arg(argc, argv, "threads",  "1"));
    int         n_iters  = atoi(get_arg(argc, argv, "iters",    "10"));
    int         explicit_model_seed = atoi(get_arg(argc, argv, "model_seed", "-1"));
    float       sigma_true = atof(get_arg(argc, argv, "sigma",  "1.0"));
    const char* csv_path = get_arg(argc, argv, "csv", "bench/results/bench_results.csv");

    bool xbart = (strcmp(mode, "xbart") == 0);
    int  n_burnin  = atoi(get_arg(argc, argv, "burnin",  xbart ? "15"   : "200"));
    int  n_samples = atoi(get_arg(argc, argv, "samples", xbart ? "25"   : "1000"));

    int n_total = n_train + n_test;

    // BARTConfig — independent of seed, set once outside the loop
    float tau = (0.5f * 0.5f) / num_trees;
    bart::BARTConfig cfg;
    cfg.num_trees      = num_trees;
    cfg.leaf_prior_var = tau;
    cfg.sigma2_shape   = 3.0f;
    cfg.sigma2_scale   = sigma_true * sigma_true;
    if (xbart) cfg.p_eval = (int)std::sqrt((float)p);
    cfg.num_threads = num_threads;

    const char* tag = xbart ? "gfr-v15-depth8" : "v15-depth8";
    printf("faststochtree %s benchmark (%s)\n", xbart ? "XBART" : "BART", tag);
    printf("  n_train=%d  n_test=%d  p=%d  trees=%d\n",
           n_train, n_test, p, num_trees);
    if (n_iters > 1)
        printf("  burnin=%d  samples=%d  iters=%d  sigma_true=%.2f\n\n",
               n_burnin, n_samples, n_iters, sigma_true);
    else
        printf("  burnin=%d  samples=%d  sigma_true=%.2f\n",
               n_burnin, n_samples, sigma_true);

    // Create parent directory if needed, then open CSV
    {
        std::string p(csv_path);
        auto slash = p.rfind('/');
        if (slash != std::string::npos) {
            std::string cmd = "mkdir -p \"" + p.substr(0, slash) + "\"";
            (void)system(cmd.c_str());
        }
    }
    FILE* csv = fopen(csv_path, "a");
    if (!csv) {
        fprintf(stderr, "WARNING: could not open CSV file for writing: %s\n", csv_path);
    } else {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "tag,timestamp,mode,n_train,p,trees,burnin,samples,threads,iter,n_iters,time_ms,rmse,sigma_post\n");
    }

    double total_ms = 0, total_rmse = 0, total_sigma = 0;

    for (int iter = 0; iter < n_iters; iter++) {
        // Each iteration gets its own independent seeds
        int data_seed  = seed + iter;
        int model_seed = (explicit_model_seed >= 0) ? explicit_model_seed
                                                    : seed + n_iters + iter;

        // Simulate data: y = sin(2*pi*x1) + N(0, sigma_true^2)
        bart::RNG data_rng(data_seed);
        std::vector<float> X(n_total * p), y_all(n_total);
        for (int i = 0; i < n_total; i++) {
            for (int j = 0; j < p; j++) X[i * p + j] = data_rng.uniform();
            float signal = std::sin(2.f * (float)M_PI * X[i * p + 0]);
            y_all[i] = signal + sigma_true * data_rng.normal();
        }

        bart::RNG model_rng(model_seed);
        const float* X_test = X.data() + n_train * p;

        if (n_iters > 1) { printf("iter %2d/%d: sampling...", iter + 1, n_iters); }
        else             { printf("Running %d burnin iterations + %d samples...", n_burnin, n_samples); }
        fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();
        auto result = xbart
            ? bart::run_xbart(X.data(), y_all.data(), n_train, p,
                              X_test, n_test, cfg, n_burnin, n_samples, model_rng)
            : bart::run_bart (X.data(), y_all.data(), n_train, p,
                              X_test, n_test, cfg, n_burnin, n_samples, model_rng);
        auto t1 = std::chrono::steady_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // Posterior mean over test samples
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
        double sigma_post = std::sqrt(sigma2_post);

        if (n_iters > 1) {
            printf("  %6ld ms  RMSE=%.4f  sigma=%.4f\n", ms, rmse, sigma_post);
        } else {
            printf(" done in %ld ms\n\n", ms);
            printf("Test RMSE:       %.4f  (sigma_true=%.4f)\n", rmse, sigma_true);
            printf("Posterior sigma: %.4f\n", sigma_post);
            printf("Total time:      %ld ms\n", ms);
        }

        if (csv) {
            char ts[32];
            std::time_t now = std::time(nullptr);
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
            fprintf(csv, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%.4f,%.4f\n",
                    tag, ts, xbart ? "xbart" : "bart",
                    n_train, p, num_trees, n_burnin, n_samples, num_threads,
                    iter + 1, n_iters, ms, rmse, sigma_post);
            fflush(csv);
        }

        total_ms    += ms;
        total_rmse  += rmse;
        total_sigma += sigma_post;
    }

    if (csv) fclose(csv);

    if (n_iters > 1) {
        printf("\nAverage over %d iterations:\n", n_iters);
        printf("  time:  %.0f ms\n",  total_ms    / n_iters);
        printf("  RMSE:  %.4f\n",     total_rmse  / n_iters);
        printf("  sigma: %.4f\n",     total_sigma / n_iters);
    }

    return 0;
}
