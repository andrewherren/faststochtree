/*
 * stochtree BART/XBART benchmark — apples-to-apples comparison with faststochtree.
 *
 * Identical simulated data:  y = sin(2π·x₁) + N(0, σ²),  X ~ U[0,1]^(n×p),
 *   seed 12345, float precision to match faststochtree's RNG sequence exactly.
 *
 * Matched hyperparameters:
 *   T=200, alpha=0.95, beta=2.0, min_samples_leaf=5, max_depth=6,
 *   tau = 0.25/T  (leaf prior variance N(0,tau)),
 *   sigma² ~ IG(nu/2, nu·lambda/2)  where nu=3, lambda=sigma_true².
 *
 * Modes (--mode bart|xbart):
 *   bart  — MCMC sampler, defaults: burnin=200  samples=1000  threads=-1
 *   xbart — GFR sampler,  defaults: burnin=15   samples=25    threads=4
 *             sqrt(p) feature subsampling per node (matches faststochtree gfr-v5+)
 *
 * Documented differences from faststochtree:
 *   - stochtree uses double precision; faststochtree uses float + uint8 quantization
 *   - leaf scale is fixed (faststochtree never samples tau)
 *   - stochtree's tree representation supports arbitrary depth (we cap at 6)
 *   - GFR cutpoint grid: stochtree uses --cutpoint_grid (default 100) discrete
 *     candidates; faststochtree uses 256-bin uint8 histograms
 *
 * Output format matches bench/main.cpp for easy side-by-side comparison.
 */

#include <stochtree/container.h>
#include <stochtree/data.h>
#include <stochtree/leaf_model.h>
#include <stochtree/tree_sampler.h>
#include <stochtree/variance_model.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

using namespace StochTree;

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
    const char* mode = get_arg(argc, argv, "mode", "bart");
    bool xbart = (strcmp(mode, "xbart") == 0);

    int    n_train    = atoi(get_arg(argc, argv, "n_train",  "50000"));
    int    n_test     = atoi(get_arg(argc, argv, "n_test",   "500"));
    int    p          = atoi(get_arg(argc, argv, "p",        "50"));
    int    num_trees  = atoi(get_arg(argc, argv, "trees",    "200"));
    int    seed       = atoi(get_arg(argc, argv, "seed",     "12345"));
    double sigma_true = atof(get_arg(argc, argv, "sigma",    "1.0"));

    // Mode-specific defaults mirror faststochtree bench/main.cpp
    int n_burnin    = atoi(get_arg(argc, argv, "burnin",  xbart ? "15"   : "200"));
    int n_samples   = atoi(get_arg(argc, argv, "samples", xbart ? "25"   : "1000"));
    int num_threads = atoi(get_arg(argc, argv, "threads", xbart ? "4"    : "-1"));
    // GFR cutpoint grid: stochtree's analogue of faststochtree's 256-bin histograms.
    // faststochtree uses all 255 uint8 cutpoints; 100 is stochtree's default.
    int cutpoint_grid = atoi(get_arg(argc, argv, "cutpoint_grid", "100"));

    int n_total = n_train + n_test;

    printf("stochtree %s benchmark (%s)\n", xbart ? "XBART" : "BART", xbart ? "GFR" : "MCMC");
    printf("  n_train=%d  n_test=%d  p=%d  trees=%d\n", n_train, n_test, p, num_trees);
    printf("  burnin=%d  samples=%d  sigma_true=%.2f\n", n_burnin, n_samples, sigma_true);
    if (xbart) printf("  threads=%d  cutpoint_grid=%d\n", num_threads, cutpoint_grid);

    // ── Generate data ─────────────────────────────────────────────────────────
    // Use float distributions to match faststochtree's RNG sequence exactly.
    std::mt19937 data_rng((unsigned)seed);
    std::uniform_real_distribution<float> unif(0.f, 1.f);
    std::normal_distribution<float>       norm(0.f, 1.f);

    std::vector<double> X(n_total * p);
    std::vector<double> y_all(n_total);

    for (int i = 0; i < n_total; i++) {
        for (int j = 0; j < p; j++) X[i * p + j] = (double)unif(data_rng);
        float signal = std::sin(2.f * (float)M_PI * (float)X[i * p + 0]);
        y_all[i] = (double)(signal + (float)sigma_true * norm(data_rng));
    }

    // ── Hyperparameters ───────────────────────────────────────────────────────
    double alpha            = 0.95;
    double beta             = 2.0;
    int    min_samples_leaf = 5;
    int    max_depth        = 6;
    double tau              = (0.5 * 0.5) / (double)num_trees;
    // sigma2 ~ IG(nu/2, nu*lambda/2).
    // stochtree's GlobalHomoskedasticVarianceModel samples IG(a + n/2, b + RSS/2),
    // so a = nu/2 = 1.5,  b = nu*lambda/2 = 1.5*sigma_true^2.
    double a_global = 1.5;
    double b_global = 1.5 * sigma_true * sigma_true;
    // sqrt(p) feature subsampling per node for xbart (matches faststochtree gfr-v5+)
    int num_features_subsample = xbart ? (int)std::sqrt((double)p) : p;

    // ── Datasets ──────────────────────────────────────────────────────────────
    ForestDataset train_dataset;
    train_dataset.AddCovariates(X.data(), n_train, p, /*row_major=*/true);

    ForestDataset test_dataset;
    test_dataset.AddCovariates(X.data() + (size_t)n_train * p, n_test, p, /*row_major=*/true);

    // Outcome / residual — no standardization (raw scale matches faststochtree)
    ColumnVector residual(y_all.data(), n_train);

    // ── Forest ────────────────────────────────────────────────────────────────
    TreeEnsemble    active_forest(num_trees, 1, /*is_leaf_constant=*/true, /*exponentiated=*/false);
    ForestContainer forest_samples(num_trees, 1, /*is_leaf_constant=*/true, /*exponentiated=*/false);

    double y_bar = 0.0;
    for (int i = 0; i < n_train; i++) y_bar += y_all[i];
    y_bar /= n_train;
    active_forest.SetLeafValue(y_bar / num_trees);

    // ── Tracker + prior ───────────────────────────────────────────────────────
    std::vector<FeatureType> feature_types(p, FeatureType::kNumeric);
    std::vector<double>      var_weights(p, 1.0 / p);

    ForestTracker tracker(train_dataset.GetCovariates(), feature_types, num_trees, n_train);
    TreePrior     tree_prior(alpha, beta, min_samples_leaf, max_depth);

    // NOTE: UpdateResidualEntireForest has a running-sum bug (tree_pred and pred_value
    // are never reset between outer-loop iterations), which is harmless only when y_bar=0
    // (stochtree's api_debug.cpp centers the outcome first via OutcomeOffsetScale).
    // We work on raw scale, so we must use UpdatePredictions then UpdateResidualNewOutcome.
    tracker.UpdatePredictions(&active_forest, train_dataset);
    UpdateResidualNewOutcome(tracker, residual);

    // ── Leaf model (constant Gaussian, tau fixed) ─────────────────────────────
    Eigen::MatrixXd  leaf_scale_matrix = Eigen::MatrixXd::Zero(1, 1);
    LeafModelVariant leaf_model_v = leafModelFactory(
        kConstantLeafGaussian, tau, leaf_scale_matrix, 0.0, 0.0);
    auto& leaf_model = std::get<GaussianConstantLeafModel>(leaf_model_v);

    // ── Variance model ────────────────────────────────────────────────────────
    GlobalHomoskedasticVarianceModel global_var_model;
    double global_variance = sigma_true * sigma_true;

    std::vector<int> sweep_indices(num_trees);
    std::iota(sweep_indices.begin(), sweep_indices.end(), 0);

    // ── Sampling ──────────────────────────────────────────────────────────────
    std::mt19937 model_rng((unsigned)(seed + 1));

    auto t0 = std::chrono::steady_clock::now();
    printf("Running %d burnin iterations + %d samples...", n_burnin, n_samples);
    fflush(stdout);

    for (int i = 0; i < n_burnin; i++) {
        if (xbart) {
            GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
                active_forest, tracker, forest_samples, leaf_model,
                train_dataset, residual, tree_prior, model_rng,
                var_weights, sweep_indices, global_variance,
                feature_types, cutpoint_grid,
                /*keep_forest=*/false, /*pre_initialized=*/true, /*backfitting=*/true,
                num_features_subsample, num_threads);
        } else {
            MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
                active_forest, tracker, forest_samples, leaf_model,
                train_dataset, residual, tree_prior, model_rng,
                var_weights, sweep_indices, global_variance,
                /*keep_forest=*/false, /*pre_initialized=*/true, /*backfitting=*/true,
                num_threads);
        }
        global_variance = global_var_model.SampleVarianceParameter(
            residual.GetData(), a_global, b_global, model_rng);
    }

    for (int i = 0; i < n_samples; i++) {
        if (xbart) {
            GFRSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
                active_forest, tracker, forest_samples, leaf_model,
                train_dataset, residual, tree_prior, model_rng,
                var_weights, sweep_indices, global_variance,
                feature_types, cutpoint_grid,
                /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
                num_features_subsample, num_threads);
        } else {
            MCMCSampleOneIter<GaussianConstantLeafModel, GaussianConstantSuffStat>(
                active_forest, tracker, forest_samples, leaf_model,
                train_dataset, residual, tree_prior, model_rng,
                var_weights, sweep_indices, global_variance,
                /*keep_forest=*/true, /*pre_initialized=*/true, /*backfitting=*/true,
                num_threads);
        }
        global_variance = global_var_model.SampleVarianceParameter(
            residual.GetData(), a_global, b_global, model_rng);
    }

    auto t1 = std::chrono::steady_clock::now();
    long ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
    printf(" done in %ld ms\n", ms);

    // ── Test RMSE ─────────────────────────────────────────────────────────────
    // Predict returns flat vector size n_test * n_samples, sample-major layout:
    //   pred[s * n_test + i] = sample s, observation i.
    std::vector<double> preds_flat = forest_samples.Predict(test_dataset);

    std::vector<double> post_mean(n_test, 0.0);
    for (int s = 0; s < n_samples; s++)
        for (int i = 0; i < n_test; i++)
            post_mean[i] += preds_flat[(size_t)s * n_test + i] / n_samples;

    double rss = 0.0;
    const double* y_test = y_all.data() + n_train;
    for (int i = 0; i < n_test; i++) {
        double err = y_test[i] - post_mean[i];
        rss += err * err;
    }
    double rmse = std::sqrt(rss / n_test);

    printf("\nTest RMSE:       %.4f  (sigma_true=%.4f)\n", rmse, sigma_true);
    printf("Posterior sigma: %.4f\n", std::sqrt(global_variance));
    printf("Total time:      %ld ms\n", ms);

    return 0;
}
