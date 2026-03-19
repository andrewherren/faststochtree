#include "faststochtree/mcmc.hpp"
#include <cmath>
#include <unordered_map>

namespace bart {

// -----------------------------------------------------------------------
// Helpers: gather per-node statistics by re-traversing all observations.
// Deliberately O(n * depth) — not optimized until v7.
// -----------------------------------------------------------------------

static void node_stats(const Tree& tree, int target_k,
                       const QuantizedX& Xq, const float* resid, int n,
                       float& sum_y, int& count) {
    sum_y = 0.f; count = 0;
    for (int i = 0; i < n; i++) {
        if (tree.traverse(Xq.data.data(), i, Xq.n) == target_k) {
            sum_y += resid[i];
            count++;
        }
    }
}

// -----------------------------------------------------------------------
// Grow proposal
// -----------------------------------------------------------------------

static void propose_grow(Tree& tree, const QuantizedX& Xq, const float* resid,
                         int n, float sigma2,
                         const BARTConfig& cfg, RNG& rng,
                         float prob_grow) {
    auto ls         = tree.leaves();
    int  num_leaves = (int)ls.size();

    int lk = ls[rng.randint(0, num_leaves)];
    if (lk > tree.half_size) return;  // bottom-level leaf, can't grow
    int leaf_depth = Tree::depth_of(lk);
    int var        = rng.randint(0, Xq.p);

    uint8_t var_min = 255, var_max = 0;
    for (int i = 0; i < n; i++) {
        if (tree.traverse(Xq.data.data(), i, Xq.n) == lk) {
            uint8_t v = Xq.at(i, var);
            if (v < var_min) var_min = v;
            if (v > var_max) var_max = v;
        }
    }
    if (var_max <= var_min) return;

    uint8_t threshold = (uint8_t)rng.randint((int)var_min, (int)var_max);

    float left_sum = 0.f, right_sum = 0.f, node_sum = 0.f;
    int   left_n   = 0,   right_n   = 0,   node_n   = 0;
    for (int i = 0; i < n; i++) {
        if (tree.traverse(Xq.data.data(), i, Xq.n) == lk) {
            node_sum += resid[i]; node_n++;
            if (Xq.at(i, var) <= threshold) { left_sum  += resid[i]; left_n++;  }
            else                             { right_sum += resid[i]; right_n++; }
        }
    }

    if (left_n < cfg.min_samples_leaf || right_n < cfg.min_samples_leaf) return;

    float split_log_ml    = leaf_log_ml(left_sum,  left_n,  sigma2, cfg.leaf_prior_var)
                          + leaf_log_ml(right_sum, right_n, sigma2, cfg.leaf_prior_var);
    float no_split_log_ml = leaf_log_ml(node_sum,  node_n,  sigma2, cfg.leaf_prior_var);

    float pg  = cfg.alpha / std::pow(1.f + leaf_depth,     cfg.beta);
    float pgl = cfg.alpha / std::pow(1.f + leaf_depth + 1, cfg.beta);
    float pgr = pgl;

    bool  grow_after     = (left_n >= 2 * cfg.min_samples_leaf ||
                            right_n >= 2 * cfg.min_samples_leaf);
    float prob_prune_new = grow_after ? 0.5f : 1.0f;

    int num_leaf_parents_new = (int)tree.leaf_parents().size() + 1;

    float log_mh = std::log(pg) + std::log(1.0f - pgl) + std::log(1.0f - pgr)
                 - std::log(1.0f - pg)
                 + std::log(prob_prune_new) - std::log(prob_grow)
                 + std::log(1.0f / num_leaf_parents_new)
                 - std::log(1.0f / num_leaves)
                 + split_log_ml - no_split_log_ml;

    if (log_mh > 0.f) log_mh = 0.f;

    if (std::log(rng.uniform()) <= log_mh)
        tree.grow(lk, var, threshold);
}

// -----------------------------------------------------------------------
// Prune proposal
// -----------------------------------------------------------------------

static void propose_prune(Tree& tree, const QuantizedX& Xq, const float* resid,
                          int n, float sigma2,
                          const BARTConfig& cfg, RNG& rng,
                          float prob_prune) {
    auto lps        = tree.leaf_parents();
    int  num_lp     = (int)lps.size();
    int  num_leaves = (int)tree.leaves().size();

    int pk          = lps[rng.randint(0, num_lp)];
    int left_child  = 2 * pk;
    int right_child = 2 * pk + 1;
    int leaf_depth  = Tree::depth_of(pk);

    float left_sum, right_sum;
    int   left_n,   right_n;
    node_stats(tree, left_child,  Xq, resid, n, left_sum,  left_n);
    node_stats(tree, right_child, Xq, resid, n, right_sum, right_n);
    float node_sum = left_sum + right_sum;
    int   node_n   = left_n   + right_n;

    float split_log_ml    = leaf_log_ml(left_sum,  left_n,  sigma2, cfg.leaf_prior_var)
                          + leaf_log_ml(right_sum, right_n, sigma2, cfg.leaf_prior_var);
    float no_split_log_ml = leaf_log_ml(node_sum,  node_n,  sigma2, cfg.leaf_prior_var);

    float pg  = cfg.alpha / std::pow(1.f + leaf_depth,     cfg.beta);
    float pgl = cfg.alpha / std::pow(1.f + leaf_depth + 1, cfg.beta);
    float pgr = pgl;

    bool  prune_after   = (num_lp >= 2);
    float prob_grow_new = prune_after ? 0.5f : 1.0f;

    float log_mh = std::log(1.0f - pg) - std::log(pg)
                 - std::log(1.0f - pgl) - std::log(1.0f - pgr)
                 + std::log(prob_prune) - std::log(prob_grow_new)
                 + std::log(1.0f / (num_leaves - 1))
                 - std::log(1.0f / num_lp)
                 + no_split_log_ml - split_log_ml;

    if (log_mh > 0.f) log_mh = 0.f;

    if (std::log(rng.uniform()) <= log_mh)
        tree.prune(pk);
}

// -----------------------------------------------------------------------
// One tree update step: decide grow vs prune, then execute
// -----------------------------------------------------------------------

static void propose_move(Tree& tree, const QuantizedX& Xq, const float* resid,
                         int n, float sigma2,
                         const BARTConfig& cfg, RNG& rng) {
    auto ls  = tree.leaves();
    auto lps = tree.leaf_parents();

    bool grow_possible = false;
    for (int lk : ls) {
        if (lk > tree.half_size) continue;  // bottom-level, can't grow
        int cnt = 0;
        for (int i = 0; i < n; i++)
            if (tree.traverse(Xq.data.data(), i, Xq.n) == lk) cnt++;
        if (cnt >= 2 * cfg.min_samples_leaf) { grow_possible = true; break; }
    }
    bool prune_possible = !lps.empty();

    if (!grow_possible && !prune_possible) return;

    float prob_grow  = grow_possible  ? (prune_possible ? 0.5f : 1.0f) : 0.0f;
    float prob_prune = prune_possible ? (grow_possible  ? 0.5f : 1.0f) : 0.0f;

    if (rng.uniform() < prob_grow)
        propose_grow(tree,  Xq, resid, n, sigma2, cfg, rng, prob_grow);
    else
        propose_prune(tree, Xq, resid, n, sigma2, cfg, rng, prob_prune);
}

// -----------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------

void sample_sigma2(const float* resid, int n, float& sigma2,
                   const BARTConfig& cfg, RNG& rng) {
    float rss = 0.f;
    for (int i = 0; i < n; i++) rss += resid[i] * resid[i];
    float shape = (cfg.sigma2_shape + n) / 2.0f;
    float scale = (cfg.sigma2_shape * cfg.sigma2_scale + rss) / 2.0f;
    sigma2 = scale / rng.gamma(shape);
}

void sample_leaves(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, float sigma2, const BARTConfig& cfg, RNG& rng) {
    float tau = cfg.leaf_prior_var;

    std::unordered_map<int, float> sum_map;
    std::unordered_map<int, int>   cnt_map;
    for (int i = 0; i < n; i++) {
        int k = tree.traverse(Xq.data.data(), i, Xq.n);
        sum_map[k] += resid[i];
        cnt_map[k]++;
    }

    for (auto& [k, sum_y] : sum_map) {
        int   cnt       = cnt_map[k];
        float post_mean = (tau * sum_y) / (cnt * tau + sigma2);
        float post_var  = (tau * sigma2) / (cnt * tau + sigma2);
        tree.leaf_value[k] = post_mean + std::sqrt(post_var) * rng.normal();
    }
}

void init_state(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees;
    state.trees.clear();
    for (int t = 0; t < T; t++) state.trees.emplace_back(cfg.tree_depth);
    state.pred.assign(T, std::vector<float>(state.n, 0.f));
    state.residual.assign(state.y, state.y + state.n);
    state.sigma2 = 1.f;
    (void)rng;
}

void mcmc_sweep(BARTState& s, const BARTConfig& cfg, RNG& rng) {
    for (int t = 0; t < cfg.num_trees; t++) {
        for (int i = 0; i < s.n; i++) s.residual[i] += s.pred[t][i];

        propose_move(s.trees[t], s.Xq, s.residual.data(),
                     s.n, s.sigma2, cfg, rng);

        sample_leaves(s.trees[t], s.Xq, s.residual.data(),
                      s.n, s.sigma2, cfg, rng);

        for (int i = 0; i < s.n; i++)
            s.pred[t][i] = s.trees[t].leaf_value[
                s.trees[t].traverse(s.Xq.data.data(), i, s.n)];
        for (int i = 0; i < s.n; i++) s.residual[i] -= s.pred[t][i];
    }
    sample_sigma2(s.residual.data(), s.n, s.sigma2, cfg, rng);
}

} // namespace bart
