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

// DGP:  mu(x) = 2*sin(pi*x1),  tau(x) = 3*x2,  pi(x) = 0.3 + 0.4*x1
//       z ~ Bernoulli(pi(x)),  y = mu(x) + tau(x)*z + N(0, sigma^2)
struct BCFBenchData {
    int n, p;
    std::vector<float> X;          // [n × p]
    std::vector<float> X_mu;       // [n × (p+1)]: X with pi_hat appended
    std::vector<float> z;
    std::vector<float> pi_hat;     // true propensity scores
    std::vector<float> tau_true;   // 3 * x2
    std::vector<float> y_scaled;
    float y_mean, y_sd;
};

static BCFBenchData simulate(int n, int p, bart::RNG& rng, float sigma) {
    BCFBenchData d;
    d.n = n; d.p = p;
    d.X.resize(n * p);
    d.z.resize(n);
    d.pi_hat.resize(n);
    d.tau_true.resize(n);
    std::vector<float> y(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) d.X[i * p + j] = rng.uniform();
        float x1 = d.X[i * p + 0], x2 = d.X[i * p + 1];
        float mu  = 2.0f * std::sin((float)M_PI * x1);
        float tau = 3.0f * x2;
        float pi  = 0.3f + 0.4f * x1;
        d.pi_hat[i]  = pi;
        d.tau_true[i] = tau;
        d.z[i] = (rng.uniform() < pi) ? 1.0f : 0.0f;
        y[i]   = mu + tau * d.z[i] + sigma * rng.normal();
    }

    double sum = 0.0, sum2 = 0.0;
    for (float yi : y) { sum += yi; sum2 += (double)yi * yi; }
    d.y_mean = (float)(sum / n);
    d.y_sd   = (float)std::sqrt(sum2 / n - (double)d.y_mean * d.y_mean);
    if (d.y_sd < 1e-6f) d.y_sd = 1.0f;
    d.y_scaled.resize(n);
    for (int i = 0; i < n; i++)
        d.y_scaled[i] = (y[i] - d.y_mean) / d.y_sd;

    // Build X_mu = [X | pi_hat]
    int p_mu = p + 1;
    d.X_mu.resize(n * p_mu);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) d.X_mu[i * p_mu + j] = d.X[i * p + j];
        d.X_mu[i * p_mu + p] = d.pi_hat[i];
    }

    return d;
}

int main(int argc, char** argv) {
    const char* mode      = get_arg(argc, argv, "mode",       "bcf");
    int   n_train         = atoi(get_arg(argc, argv, "n_train",    "5000"));
    int   n_test          = atoi(get_arg(argc, argv, "n_test",     "500"));
    int   p               = atoi(get_arg(argc, argv, "p",          "10"));
    int   mu_trees        = atoi(get_arg(argc, argv, "mu_trees",   "50"));
    int   tau_trees       = atoi(get_arg(argc, argv, "tau_trees",  "20"));
    int   seed            = atoi(get_arg(argc, argv, "seed",       "42"));
    int   num_threads     = atoi(get_arg(argc, argv, "threads",    "1"));
    int   n_iters         = atoi(get_arg(argc, argv, "iters",      "5"));
    int   explicit_model_seed = atoi(get_arg(argc, argv, "model_seed", "-1"));
    float sigma_true      = atof(get_arg(argc, argv, "sigma",      "1.0"));
    const char* csv_path  = get_arg(argc, argv, "csv", "bench/results/bcf_bench_results.csv");

    bool xbcf      = (strcmp(mode, "xbcf") == 0);
    bool warmstart = (strcmp(mode, "warmstart_bcf") == 0);

    int n_burnin    = atoi(get_arg(argc, argv, "burnin",      xbcf ? "20"  : "100"));
    int n_samples   = atoi(get_arg(argc, argv, "samples",     xbcf ? "25"  : "200"));
    int n_gfr_burnin = atoi(get_arg(argc, argv, "gfr_burnin", "20"));

    bart::BCFConfig cfg;
    cfg.mu_config.num_trees      = mu_trees;
    cfg.mu_config.leaf_prior_var = 0.25f / mu_trees;
    cfg.mu_config.num_threads    = num_threads;
    cfg.tau_config.num_trees     = tau_trees;
    cfg.tau_config.leaf_prior_var = 0.25f / tau_trees;

    const char* tag = warmstart ? "warmstart_bcf" : (xbcf ? "xbcf" : "bcf");
    printf("faststochtree BCF benchmark (%s mode)\n", tag);
    printf("  n_train=%d  n_test=%d  p=%d  mu_trees=%d  tau_trees=%d\n",
           n_train, n_test, p, mu_trees, tau_trees);
    if (warmstart)
        printf("  gfr_burnin=%d  mcmc_samples=%d  iters=%d  sigma_true=%.2f\n\n",
               n_gfr_burnin, n_samples, n_iters, sigma_true);
    else if (n_iters > 1)
        printf("  burnin=%d  samples=%d  iters=%d  sigma_true=%.2f\n\n",
               n_burnin, n_samples, n_iters, sigma_true);
    else
        printf("  burnin=%d  samples=%d  sigma_true=%.2f\n",
               n_burnin, n_samples, sigma_true);

    {
        std::string sp(csv_path);
        auto slash = sp.rfind('/');
        if (slash != std::string::npos) {
            std::string cmd = "mkdir -p \"" + sp.substr(0, slash) + "\"";
            (void)system(cmd.c_str());
        }
    }
    FILE* csv = fopen(csv_path, "a");
    if (!csv) {
        fprintf(stderr, "WARNING: could not open CSV file: %s\n", csv_path);
    } else {
        fseek(csv, 0, SEEK_END);
        if (ftell(csv) == 0)
            fprintf(csv, "tag,timestamp,mode,n_train,p,mu_trees,tau_trees,burnin,samples,threads,iter,n_iters,time_ms,cate_rmse,avg_tau,sigma_post\n");
    }

    double total_ms = 0, total_cate_rmse = 0, total_avg_tau = 0, total_sigma = 0;

    for (int iter = 0; iter < n_iters; iter++) {
        int data_seed  = seed + iter;
        int model_seed = (explicit_model_seed >= 0) ? explicit_model_seed
                                                    : seed + n_iters + iter;

        bart::RNG data_rng(data_seed);
        auto train = simulate(n_train, p, data_rng, sigma_true);
        auto test  = simulate(n_test,  p, data_rng, sigma_true);

        bart::RNG model_rng(model_seed);

        if (n_iters > 1) { printf("iter %2d/%d: sampling...", iter + 1, n_iters); }
        else             { printf("Running %s: burnin=%d samples=%d ...",
                                  tag, warmstart ? n_gfr_burnin : n_burnin, n_samples); }
        fflush(stdout);

        auto t0 = std::chrono::steady_clock::now();

        bart::BCFResult result;
        if (warmstart) {
            result = bart::run_warmstart_bcf(
                train.X_mu.data(), train.y_scaled.data(), train.z.data(),
                n_train, p + 1,
                train.X.data(), p,
                test.X_mu.data(), test.X.data(), n_test,
                cfg, n_gfr_burnin, 0, n_samples, model_rng);
        } else if (xbcf) {
            result = bart::run_xbcf(
                train.X_mu.data(), train.y_scaled.data(), train.z.data(),
                n_train, p + 1,
                train.X.data(), p,
                test.X_mu.data(), test.X.data(), n_test,
                cfg, n_burnin, n_samples, model_rng);
        } else {
            result = bart::run_bcf(
                train.X_mu.data(), train.y_scaled.data(), train.z.data(),
                n_train, p + 1,
                train.X.data(), p,
                test.X_mu.data(), test.X.data(), n_test,
                cfg, n_burnin, n_samples, model_rng);
        }

        auto t1 = std::chrono::steady_clock::now();
        long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // Posterior mean CATE on test set (back-scale from standardised y)
        int ns = (int)result.tau_test.size();
        std::vector<double> tau_hat(n_test, 0.0);
        for (const auto& samp : result.tau_test)
            for (int i = 0; i < n_test; i++)
                tau_hat[i] += samp[i] / ns;
        for (double& v : tau_hat) v *= train.y_sd;

        double rss = 0.0, sum_tau = 0.0;
        for (int i = 0; i < n_test; i++) {
            double err = tau_hat[i] - (double)test.tau_true[i];
            rss     += err * err;
            sum_tau += tau_hat[i];
        }
        double cate_rmse = std::sqrt(rss / n_test);
        double avg_tau   = sum_tau / n_test;

        double sigma2_post = 0.0;
        for (float s2 : result.sigma2_samples) sigma2_post += s2;
        sigma2_post /= result.sigma2_samples.size();
        double sigma_post = std::sqrt(sigma2_post) * train.y_sd;

        if (n_iters > 1) {
            printf("  %6ld ms  CATE RMSE=%.4f  avg_tau=%.4f  sigma=%.4f\n",
                   ms, cate_rmse, avg_tau, sigma_post);
        } else {
            printf(" done in %ld ms\n\n", ms);
            printf("CATE RMSE:       %.4f  (null-model ~0.87, true E[tau]=1.50)\n", cate_rmse);
            printf("Mean tau_hat:    %.4f  (true=1.50)\n", avg_tau);
            printf("Posterior sigma: %.4f  (true=%.4f)\n", sigma_post, sigma_true);
            printf("Total time:      %ld ms\n", ms);
        }

        if (csv) {
            char ts[32];
            std::time_t now = std::time(nullptr);
            std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", std::gmtime(&now));
            fprintf(csv, "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%ld,%.4f,%.4f,%.4f\n",
                    tag, ts, tag,
                    n_train, p, mu_trees, tau_trees,
                    warmstart ? n_gfr_burnin : n_burnin, n_samples,
                    num_threads, iter + 1, n_iters,
                    ms, cate_rmse, avg_tau, sigma_post);
            fflush(csv);
        }

        total_ms        += ms;
        total_cate_rmse += cate_rmse;
        total_avg_tau   += avg_tau;
        total_sigma     += sigma_post;
    }

    if (csv) fclose(csv);

    if (n_iters > 1) {
        printf("\nAverage over %d iterations:\n", n_iters);
        printf("  time:      %.0f ms\n",  total_ms        / n_iters);
        printf("  CATE RMSE: %.4f\n",     total_cate_rmse / n_iters);
        printf("  avg_tau:   %.4f\n",     total_avg_tau   / n_iters);
        printf("  sigma:     %.4f\n",     total_sigma     / n_iters);
    }

    return 0;
}
