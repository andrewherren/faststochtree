#include "cpp11.hpp"
#include "faststochtree/sampler.hpp"
#include <vector>

using namespace cpp11;

// ── helpers ───────────────────────────────────────────────────────────────────

// Convert an R numeric matrix (double, column-major) to a row-major float vector.
static std::vector<float> to_float_rowmajor(doubles_matrix<> m) {
    int n = m.nrow(), p = m.ncol();
    std::vector<float> out(n * p);
    for (int j = 0; j < p; j++)
        for (int i = 0; i < n; i++)
            out[i * p + j] = static_cast<float>(m(i, j));
    return out;
}

static std::vector<float> to_float_vec(doubles v) {
    std::vector<float> out(v.size());
    for (int i = 0; i < (int)v.size(); i++)
        out[i] = static_cast<float>(v[i]);
    return out;
}

static bart::BARTConfig bart_config_from_list(list cfg) {
    bart::BARTConfig c;
    auto get_int = [&](const char* nm, int def) -> int {
        if (cfg.names().size() == 0) return def;
        for (int i = 0; i < (int)cfg.size(); i++)
            if (std::string(cfg.names()[i]) == nm)
                return as_cpp<int>(cfg[i]);
        return def;
    };
    auto get_dbl = [&](const char* nm, float def) -> float {
        if (cfg.names().size() == 0) return def;
        for (int i = 0; i < (int)cfg.size(); i++)
            if (std::string(cfg.names()[i]) == nm)
                return static_cast<float>(as_cpp<double>(cfg[i]));
        return def;
    };
    c.num_trees        = get_int("num_trees",        200);
    c.tree_depth       = get_int("tree_depth",       6);
    c.min_samples_leaf = get_int("min_samples_leaf", 5);
    c.alpha            = get_dbl("alpha",            0.95f);
    c.beta             = get_dbl("beta",             2.0f);
    c.leaf_prior_var   = get_dbl("leaf_prior_var",   -1.0f);
    c.sigma2_shape     = get_dbl("sigma2_shape",     3.0f);
    c.sigma2_scale     = get_dbl("sigma2_scale",     1.0f);
    c.p_eval           = get_int("p_eval",           0);
    c.num_threads      = get_int("num_threads",      1);
    return c;
}

// Finalizers called when the R external pointer is GC'd.
static void bart_model_finalizer(bart::BARTModel* m) { delete m; }
static void bcf_model_finalizer(bart::BCFModel* m)   { delete m; }

// BARTConfig from a named R list (shared by BART and BCF helpers).
static bart::BARTConfig bart_config_from_list(list cfg);

// ── fit_bart_cpp ──────────────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BARTModel> fit_bart_cpp(
        doubles_matrix<> X, doubles y,
        doubles_matrix<> X_test,
        int n_burnin, int n_samples, int seed,
        list config) {

    bart::BARTConfig cfg = bart_config_from_list(config);
    std::vector<float> Xf  = to_float_rowmajor(X);
    std::vector<float> yf  = to_float_vec(y);
    std::vector<float> Xtf = to_float_rowmajor(X_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BARTModel(
        bart::fit_bart(Xf.data(), yf.data(), n, p,
                       Xtf.data(), n_test, cfg,
                       n_burnin, n_samples, seed));
    return external_pointer<bart::BARTModel>(m, &bart_model_finalizer);
}

// ── fit_xbart_cpp ─────────────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BARTModel> fit_xbart_cpp(
        doubles_matrix<> X, doubles y,
        doubles_matrix<> X_test,
        int n_burnin, int n_samples, int seed,
        list config) {

    bart::BARTConfig cfg = bart_config_from_list(config);
    std::vector<float> Xf  = to_float_rowmajor(X);
    std::vector<float> yf  = to_float_vec(y);
    std::vector<float> Xtf = to_float_rowmajor(X_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BARTModel(
        bart::fit_xbart(Xf.data(), yf.data(), n, p,
                        Xtf.data(), n_test, cfg,
                        n_burnin, n_samples, seed));
    return external_pointer<bart::BARTModel>(m, &bart_model_finalizer);
}

// ── fit_warmstart_bart_cpp ────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BARTModel> fit_warmstart_bart_cpp(
        doubles_matrix<> X, doubles y,
        doubles_matrix<> X_test,
        int n_gfr_burnin, int n_mcmc_burnin, int n_samples, int seed,
        bool keep_gfr_samples,
        list config) {

    bart::BARTConfig cfg = bart_config_from_list(config);
    std::vector<float> Xf  = to_float_rowmajor(X);
    std::vector<float> yf  = to_float_vec(y);
    std::vector<float> Xtf = to_float_rowmajor(X_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BARTModel(
        bart::fit_warmstart_bart(Xf.data(), yf.data(), n, p,
                                  Xtf.data(), n_test, cfg,
                                  n_gfr_burnin, n_mcmc_burnin, n_samples,
                                  seed, keep_gfr_samples));
    return external_pointer<bart::BARTModel>(m, &bart_model_finalizer);
}

// ── predict_cpp ───────────────────────────────────────────────────────────────

[[cpp11::register]]
doubles_matrix<> predict_cpp(external_pointer<bart::BARTModel> model,
                              doubles_matrix<> X_new) {
    std::vector<float> Xf = to_float_rowmajor(X_new);
    int n_new = X_new.nrow();

    std::vector<float> out = model->predict(Xf.data(), n_new);

    // Return [n_samples × n_new] matrix
    writable::doubles_matrix<> result(model->n_samples, n_new);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < n_new; i++)
            result(s, i) = static_cast<double>(out[s * n_new + i]);
    return result;
}

// ── test_samples_cpp / sigma2_samples_cpp ────────────────────────────────────

[[cpp11::register]]
doubles_matrix<> test_samples_cpp(external_pointer<bart::BARTModel> model) {
    writable::doubles_matrix<> result(model->n_samples, model->n_test);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < model->n_test; i++)
            result(s, i) = static_cast<double>(
                model->test_samples[s * model->n_test + i]);
    return result;
}

[[cpp11::register]]
doubles sigma2_samples_cpp(external_pointer<bart::BARTModel> model) {
    writable::doubles out(model->n_samples);
    for (int i = 0; i < model->n_samples; i++)
        out[i] = static_cast<double>(model->sigma2_samples[i]);
    return out;
}

// ── BCF helpers ───────────────────────────────────────────────────────────────

static bart::BCFConfig bcf_config_from_lists(list mu_cfg_list, list tau_cfg_list) {
    bart::BCFConfig cfg;
    cfg.mu_config  = bart_config_from_list(mu_cfg_list);
    cfg.tau_config = bart_config_from_list(tau_cfg_list);
    return cfg;
}

// ── fit_bcf_cpp ───────────────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BCFModel> fit_bcf_cpp(
        doubles_matrix<> X, doubles y, doubles z, doubles pi_hat,
        doubles_matrix<> X_test, doubles pi_hat_test,
        int n_burnin, int n_samples, int seed,
        list mu_config, list tau_config) {

    bart::BCFConfig cfg = bcf_config_from_lists(mu_config, tau_config);
    std::vector<float> Xf     = to_float_rowmajor(X);
    std::vector<float> yf     = to_float_vec(y);
    std::vector<float> zf     = to_float_vec(z);
    std::vector<float> pif    = to_float_vec(pi_hat);
    std::vector<float> Xtf    = to_float_rowmajor(X_test);
    std::vector<float> pitf   = to_float_vec(pi_hat_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BCFModel(
        bart::fit_bcf(Xf.data(), yf.data(), zf.data(), pif.data(), n, p,
                      Xtf.data(), pitf.data(), n_test, cfg,
                      n_burnin, n_samples, seed));
    return external_pointer<bart::BCFModel>(m, &bcf_model_finalizer);
}

// ── fit_xbcf_cpp ──────────────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BCFModel> fit_xbcf_cpp(
        doubles_matrix<> X, doubles y, doubles z, doubles pi_hat,
        doubles_matrix<> X_test, doubles pi_hat_test,
        int n_burnin, int n_samples, int seed,
        list mu_config, list tau_config) {

    bart::BCFConfig cfg = bcf_config_from_lists(mu_config, tau_config);
    std::vector<float> Xf     = to_float_rowmajor(X);
    std::vector<float> yf     = to_float_vec(y);
    std::vector<float> zf     = to_float_vec(z);
    std::vector<float> pif    = to_float_vec(pi_hat);
    std::vector<float> Xtf    = to_float_rowmajor(X_test);
    std::vector<float> pitf   = to_float_vec(pi_hat_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BCFModel(
        bart::fit_xbcf(Xf.data(), yf.data(), zf.data(), pif.data(), n, p,
                       Xtf.data(), pitf.data(), n_test, cfg,
                       n_burnin, n_samples, seed));
    return external_pointer<bart::BCFModel>(m, &bcf_model_finalizer);
}

// ── fit_warmstart_bcf_cpp ─────────────────────────────────────────────────────

[[cpp11::register]]
external_pointer<bart::BCFModel> fit_warmstart_bcf_cpp(
        doubles_matrix<> X, doubles y, doubles z, doubles pi_hat,
        doubles_matrix<> X_test, doubles pi_hat_test,
        int n_gfr_burnin, int n_mcmc_burnin, int n_samples, int seed,
        bool keep_gfr_samples,
        list mu_config, list tau_config) {

    bart::BCFConfig cfg = bcf_config_from_lists(mu_config, tau_config);
    std::vector<float> Xf     = to_float_rowmajor(X);
    std::vector<float> yf     = to_float_vec(y);
    std::vector<float> zf     = to_float_vec(z);
    std::vector<float> pif    = to_float_vec(pi_hat);
    std::vector<float> Xtf    = to_float_rowmajor(X_test);
    std::vector<float> pitf   = to_float_vec(pi_hat_test);

    int n      = X.nrow(), p = X.ncol();
    int n_test = X_test.nrow();

    auto* m = new bart::BCFModel(
        bart::fit_warmstart_bcf(Xf.data(), yf.data(), zf.data(), pif.data(), n, p,
                                 Xtf.data(), pitf.data(), n_test, cfg,
                                 n_gfr_burnin, n_mcmc_burnin, n_samples,
                                 seed, keep_gfr_samples));
    return external_pointer<bart::BCFModel>(m, &bcf_model_finalizer);
}

// ── BCF model accessors ───────────────────────────────────────────────────────

[[cpp11::register]]
doubles_matrix<> predict_tau_cpp(external_pointer<bart::BCFModel> model,
                                  doubles_matrix<> X_new) {
    std::vector<float> Xf = to_float_rowmajor(X_new);
    int n_new = X_new.nrow();
    std::vector<float> out = model->predict_tau(Xf.data(), n_new);
    writable::doubles_matrix<> result(model->n_samples, n_new);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < n_new; i++)
            result(s, i) = static_cast<double>(out[s * n_new + i]);
    return result;
}

[[cpp11::register]]
doubles_matrix<> predict_mu_cpp(external_pointer<bart::BCFModel> model,
                                 doubles_matrix<> X_new_mu) {
    std::vector<float> Xf = to_float_rowmajor(X_new_mu);
    int n_new = X_new_mu.nrow();
    std::vector<float> out = model->predict_mu(Xf.data(), n_new);
    writable::doubles_matrix<> result(model->n_samples, n_new);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < n_new; i++)
            result(s, i) = static_cast<double>(out[s * n_new + i]);
    return result;
}

[[cpp11::register]]
doubles_matrix<> test_tau_samples_cpp(external_pointer<bart::BCFModel> model) {
    writable::doubles_matrix<> result(model->n_samples, model->n_test);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < model->n_test; i++)
            result(s, i) = static_cast<double>(
                model->test_tau[s * model->n_test + i]);
    return result;
}

[[cpp11::register]]
doubles_matrix<> test_mu_samples_cpp(external_pointer<bart::BCFModel> model) {
    writable::doubles_matrix<> result(model->n_samples, model->n_test);
    for (int s = 0; s < model->n_samples; s++)
        for (int i = 0; i < model->n_test; i++)
            result(s, i) = static_cast<double>(
                model->test_mu[s * model->n_test + i]);
    return result;
}

[[cpp11::register]]
doubles sigma2_bcf_samples_cpp(external_pointer<bart::BCFModel> model) {
    writable::doubles out(model->n_samples);
    for (int i = 0; i < model->n_samples; i++)
        out[i] = static_cast<double>(model->sigma2_samples[i]);
    return out;
}
