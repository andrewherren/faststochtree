#pragma once
#include "faststochtree/tree.hpp"
#include "faststochtree/rng.hpp"
#include <vector>
#include <cmath>

namespace bart {

struct BARTConfig {
    int    num_trees        = 200;
    int    min_samples_leaf = 5;
    double alpha            = 0.95;   // BART tree prior: P(split at depth d) = alpha/(1+d)^beta
    double beta             = 2.0;
    double leaf_prior_var   = 1.0;    // tau: leaf value prior is N(0, tau)
    double sigma2_shape     = 3.0;    // nu:     IG(nu/2, nu*lambda/2) prior on sigma2
    double sigma2_scale     = 1.0;    // lambda
};

struct BARTState {
    int n, p;
    const double* X;     // row-major covariates [n x p]
    const double* y;     // outcomes [n]

    std::vector<Tree>                trees;
    std::vector<std::vector<double>> pred;     // pred[t][i]: tree t's prediction for obs i
    std::vector<double>              residual; // partial residual
    double                           sigma2;
};

// Reduced log marginal likelihood for a Gaussian constant leaf.
// Cancels terms that are equal between split and no-split comparisons.
// See stochtree leaf_model.h for derivation.
inline double leaf_log_ml(double sum_y, int n, double sigma2, double tau) {
    return -0.5 * std::log(1.0 + tau * n / sigma2)
           + (tau * sum_y * sum_y) / (2.0 * sigma2 * (n * tau + sigma2));
}

// Sample sigma2 ~ IG(nu/2 + n/2,  nu*lambda/2 + RSS/2)
void sample_sigma2(const double* resid, int n, double& sigma2,
                   const BARTConfig& cfg, RNG& rng);

// Sample leaf values for all leaves of one tree (Gaussian posterior)
void sample_leaves(Tree& tree, const double* X, const double* resid,
                   int n, int p, double sigma2, const BARTConfig& cfg, RNG& rng);

} // namespace bart
