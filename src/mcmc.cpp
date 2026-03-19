#include "faststochtree/mcmc.hpp"
#include <cmath>
#include <limits>
#include <unordered_map>

namespace bart {

// -----------------------------------------------------------------------
// Helpers: gather per-node statistics by re-traversing all observations.
// Deliberately O(n * depth) — this is the v1 baseline, not optimized.
// -----------------------------------------------------------------------

// Gather stats (sum_y, count) for observations that land in node target_k.
static void node_stats(const Tree& tree, int target_k,
                       const double* X, const double* resid, int n, int p,
                       double& sum_y, int& count) {
    sum_y = 0.0; count = 0;
    for (int i = 0; i < n; i++) {
        if (tree.traverse(X, i, p) == target_k) {
            sum_y += resid[i];
            count++;
        }
    }
}

// -----------------------------------------------------------------------
// Grow proposal
// -----------------------------------------------------------------------

static void propose_grow(Tree& tree, const double* X, const double* resid,
                         int n, int p, double sigma2,
                         const BARTConfig& cfg, RNG& rng,
                         double prob_grow) {
    auto ls         = tree.leaves();
    int  num_leaves = (int)ls.size();

    // Pick a random leaf
    int  lk          = ls[rng.randint(0, num_leaves)];
    int  leaf_depth  = tree.nodes[lk].depth;

    // Pick a random feature
    int var = rng.randint(0, p);

    // Find the range of that feature among obs in this leaf
    double var_min =  std::numeric_limits<double>::max();
    double var_max = -std::numeric_limits<double>::max();
    for (int i = 0; i < n; i++) {
        if (tree.traverse(X, i, p) == lk) {
            double v = X[i * p + var];
            if (v < var_min) var_min = v;
            if (v > var_max) var_max = v;
        }
    }
    if (var_max <= var_min) return;  // constant feature in this leaf

    double threshold = var_min + rng.uniform() * (var_max - var_min);

    // Accumulate sufficient stats for proposed left/right nodes
    double left_sum = 0.0, right_sum = 0.0, node_sum = 0.0;
    int    left_n   = 0,   right_n   = 0,   node_n   = 0;
    for (int i = 0; i < n; i++) {
        if (tree.traverse(X, i, p) == lk) {
            node_sum += resid[i]; node_n++;
            if (X[i * p + var] <= threshold) { left_sum  += resid[i]; left_n++;  }
            else                              { right_sum += resid[i]; right_n++; }
        }
    }

    if (left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf) return;

    double split_log_ml    = leaf_log_ml(left_sum,  left_n,  sigma2, cfg.leaf_prior_var)
                           + leaf_log_ml(right_sum, right_n, sigma2, cfg.leaf_prior_var);
    double no_split_log_ml = leaf_log_ml(node_sum,  node_n,  sigma2, cfg.leaf_prior_var);

    double pg  = cfg.alpha / std::pow(1.0 + leaf_depth,     cfg.beta);
    double pgl = cfg.alpha / std::pow(1.0 + leaf_depth + 1, cfg.beta);
    double pgr = pgl;

    bool   grow_after     = (left_n >= 2 * cfg.min_samples_leaf ||
                             right_n >= 2 * cfg.min_samples_leaf);
    double prob_prune_new = grow_after ? 0.5 : 1.0;

    int num_leaf_parents_new = (int)tree.leaf_parents().size() + 1;

    double log_mh = std::log(pg) + std::log(1.0 - pgl) + std::log(1.0 - pgr)
                  - std::log(1.0 - pg)
                  + std::log(prob_prune_new) - std::log(prob_grow)
                  + std::log(1.0 / num_leaf_parents_new)
                  - std::log(1.0 / num_leaves)
                  + split_log_ml - no_split_log_ml;

    if (log_mh > 0.0) log_mh = 0.0;

    if (std::log(rng.uniform()) <= log_mh)
        tree.grow(lk, var, threshold);
}

// -----------------------------------------------------------------------
// Prune proposal
// -----------------------------------------------------------------------

static void propose_prune(Tree& tree, const double* X, const double* resid,
                          int n, int p, double sigma2,
                          const BARTConfig& cfg, RNG& rng,
                          double prob_prune) {
    auto lps        = tree.leaf_parents();
    int  num_lp     = (int)lps.size();
    int  num_leaves = (int)tree.leaves().size();

    // Pick a random leaf parent
    int pk          = lps[rng.randint(0, num_lp)];
    int left_child  = tree.nodes[pk].left;
    int right_child = tree.nodes[pk].right;
    int leaf_depth  = tree.nodes[pk].depth;

    // Sufficient stats for left, right, and combined
    double left_sum, right_sum;
    int    left_n,   right_n;
    node_stats(tree, left_child,  X, resid, n, p, left_sum,  left_n);
    node_stats(tree, right_child, X, resid, n, p, right_sum, right_n);
    double node_sum = left_sum + right_sum;
    int    node_n   = left_n   + right_n;

    double split_log_ml    = leaf_log_ml(left_sum,  left_n,  sigma2, cfg.leaf_prior_var)
                           + leaf_log_ml(right_sum, right_n, sigma2, cfg.leaf_prior_var);
    double no_split_log_ml = leaf_log_ml(node_sum,  node_n,  sigma2, cfg.leaf_prior_var);

    double pg  = cfg.alpha / std::pow(1.0 + leaf_depth,     cfg.beta);
    double pgl = cfg.alpha / std::pow(1.0 + leaf_depth + 1, cfg.beta);
    double pgr = pgl;

    bool   prune_after   = (num_lp >= 2);
    double prob_grow_new = prune_after ? 0.5 : 1.0;

    double log_mh = std::log(1.0 - pg) - std::log(pg)
                  - std::log(1.0 - pgl) - std::log(1.0 - pgr)
                  + std::log(prob_prune) - std::log(prob_grow_new)
                  + std::log(1.0 / (num_leaves - 1))
                  - std::log(1.0 / num_lp)
                  + no_split_log_ml - split_log_ml;

    if (log_mh > 0.0) log_mh = 0.0;

    if (std::log(rng.uniform()) <= log_mh)
        tree.prune(pk);
}

// -----------------------------------------------------------------------
// One tree update step: decide grow vs prune, then execute
// -----------------------------------------------------------------------

static void propose_move(Tree& tree, const double* X, const double* resid,
                         int n, int p, double sigma2,
                         const BARTConfig& cfg, RNG& rng) {
    auto ls  = tree.leaves();
    auto lps = tree.leaf_parents();

    // Grow is possible if any leaf holds enough observations
    bool grow_possible = false;
    for (int lk : ls) {
        int cnt = 0;
        for (int i = 0; i < n; i++)
            if (tree.traverse(X, i, p) == lk) cnt++;
        if (cnt >= 2 * cfg.min_samples_leaf) { grow_possible = true; break; }
    }
    bool prune_possible = !lps.empty();

    if (!grow_possible && !prune_possible) return;

    double prob_grow  = grow_possible  ? (prune_possible ? 0.5 : 1.0) : 0.0;
    double prob_prune = prune_possible ? (grow_possible  ? 0.5 : 1.0) : 0.0;

    if (rng.uniform() < prob_grow)
        propose_grow(tree,  X, resid, n, p, sigma2, cfg, rng, prob_grow);
    else
        propose_prune(tree, X, resid, n, p, sigma2, cfg, rng, prob_prune);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

void sample_sigma2(const double* resid, int n, double& sigma2,
                   const BARTConfig& cfg, RNG& rng) {
    double rss = 0.0;
    for (int i = 0; i < n; i++) rss += resid[i] * resid[i];
    double shape = (cfg.sigma2_shape + n) / 2.0;
    double scale = (cfg.sigma2_shape * cfg.sigma2_scale + rss) / 2.0;
    sigma2 = scale / rng.gamma(shape);
}

void sample_leaves(Tree& tree, const double* X, const double* resid,
                   int n, int p, double sigma2, const BARTConfig& cfg, RNG& rng) {
    double tau = cfg.leaf_prior_var;

    std::unordered_map<int, double> sum_map;
    std::unordered_map<int, int>    cnt_map;
    for (int i = 0; i < n; i++) {
        int k = tree.traverse(X, i, p);
        sum_map[k] += resid[i];
        cnt_map[k]++;
    }

    for (auto& [k, sum_y] : sum_map) {
        int    cnt       = cnt_map[k];
        double post_mean = (tau * sum_y) / (cnt * tau + sigma2);
        double post_var  = (tau * sigma2) / (cnt * tau + sigma2);
        tree.nodes[k].value = post_mean + std::sqrt(post_var) * rng.normal();
    }
}

void init_state(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees;
    state.trees.clear();
    for (int t = 0; t < T; t++) state.trees.emplace_back();
    state.pred.assign(T, std::vector<double>(state.n, 0.0));
    state.residual.assign(state.y, state.y + state.n);
    state.sigma2 = 1.0;
    (void)rng;
}

void mcmc_sweep(BARTState& s, const BARTConfig& cfg, RNG& rng) {
    for (int t = 0; t < cfg.num_trees; t++) {
        for (int i = 0; i < s.n; i++) s.residual[i] += s.pred[t][i];

        propose_move(s.trees[t], s.X, s.residual.data(),
                     s.n, s.p, s.sigma2, cfg, rng);

        sample_leaves(s.trees[t], s.X, s.residual.data(),
                      s.n, s.p, s.sigma2, cfg, rng);

        for (int i = 0; i < s.n; i++)
            s.pred[t][i] = s.trees[t].nodes[s.trees[t].traverse(s.X, i, s.p)].value;
        for (int i = 0; i < s.n; i++) s.residual[i] -= s.pred[t][i];
    }
    sample_sigma2(s.residual.data(), s.n, s.sigma2, cfg, rng);
}

} // namespace bart
