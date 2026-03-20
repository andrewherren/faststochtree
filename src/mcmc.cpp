#include "faststochtree/mcmc.hpp"
#include <cmath>
#include <cstring>
#include <vector>

namespace bart {

// -----------------------------------------------------------------------
// Grow proposal — uses cached leaf_idx; updates cache on acceptance
// -----------------------------------------------------------------------

static void propose_grow(Tree& tree, const QuantizedX& Xq, const float* resid,
                         const float* pred_off,
                         int n, float sigma2,
                         const BARTConfig& cfg, RNG& rng,
                         float prob_grow, std::vector<int>& leaf_idx,
                         std::vector<int>& leaf_counts,
                         const std::vector<int>& ls) {
    int  num_leaves = (int)ls.size();

    int lk = ls[rng.randint(0, num_leaves)];
    if (lk > tree.half_size) return;
    int leaf_depth = Tree::depth_of(lk);
    int var        = rng.randint(0, Xq.p);

    uint8_t var_min = 255, var_max = 0;
    for (int i = 0; i < n; i++) {
        if (leaf_idx[i] == lk) {
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
        if (leaf_idx[i] == lk) {
            float r = resid[i] + pred_off[i];
            node_sum += r; node_n++;
            if (Xq.at(i, var) <= threshold) { left_sum  += r; left_n++;  }
            else                             { right_sum += r; right_n++; }
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

    if (std::log(rng.uniform()) <= log_mh) {
        tree.grow(lk, var, threshold);
        // Update cache: obs at lk routed to left (2*lk) or right (2*lk+1)
        for (int i = 0; i < n; i++)
            if (leaf_idx[i] == lk)
                leaf_idx[i] = 2*lk + (Xq.at(i, var) > threshold ? 1 : 0);
        leaf_counts[2*lk]     = left_n;
        leaf_counts[2*lk + 1] = right_n;
        leaf_counts[lk]       = 0;
    }
}

// -----------------------------------------------------------------------
// Prune proposal — uses cached leaf_idx; updates cache on acceptance
// -----------------------------------------------------------------------

static void propose_prune(Tree& tree, const float* resid, const float* pred_off,
                          int n, float sigma2,
                          const BARTConfig& cfg, RNG& rng,
                          float prob_prune, std::vector<int>& leaf_idx,
                          std::vector<int>& leaf_counts,
                          const std::vector<int>& lps, int num_leaves) {
    int  num_lp     = (int)lps.size();

    int pk          = lps[rng.randint(0, num_lp)];
    int left_child  = 2 * pk;
    int right_child = 2 * pk + 1;
    int leaf_depth  = Tree::depth_of(pk);

    float left_sum = 0.f, right_sum = 0.f;
    int   left_n   = 0,   right_n   = 0;
    for (int i = 0; i < n; i++) {
        float r = resid[i] + pred_off[i];
        if      (leaf_idx[i] == left_child)  { left_sum  += r; left_n++;  }
        else if (leaf_idx[i] == right_child) { right_sum += r; right_n++; }
    }
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

    if (std::log(rng.uniform()) <= log_mh) {
        tree.prune(pk);
        // Update cache: obs at 2*pk and 2*pk+1 merge back to pk
        for (int i = 0; i < n; i++)
            if (leaf_idx[i] == left_child || leaf_idx[i] == right_child)
                leaf_idx[i] = pk;
        leaf_counts[pk]          = node_n;
        leaf_counts[left_child]  = 0;
        leaf_counts[right_child] = 0;
    }
}

// -----------------------------------------------------------------------
// One tree update step
// -----------------------------------------------------------------------

static void propose_move(Tree& tree, const QuantizedX& Xq, const float* resid,
                         const float* pred_off,
                         int n, float sigma2,
                         const BARTConfig& cfg, RNG& rng,
                         std::vector<int>& leaf_idx, std::vector<int>& leaf_counts,
                         Workspace& ws) {
    tree.leaves(ws.leaves_buf);
    tree.leaf_parents(ws.leaf_parents_buf);
    const auto& ls  = ws.leaves_buf;
    const auto& lps = ws.leaf_parents_buf;

    bool grow_possible = false;
    for (int lk : ls) {
        if (lk > tree.half_size) continue;
        if (leaf_counts[lk] >= 2 * cfg.min_samples_leaf) { grow_possible = true; break; }
    }
    bool prune_possible = !lps.empty();

    if (!grow_possible && !prune_possible) return;

    float prob_grow  = grow_possible  ? (prune_possible ? 0.5f : 1.0f) : 0.0f;
    float prob_prune = prune_possible ? (grow_possible  ? 0.5f : 1.0f) : 0.0f;

    if (rng.uniform() < prob_grow)
        propose_grow(tree,  Xq, resid, pred_off, n, sigma2, cfg, rng, prob_grow,  leaf_idx, leaf_counts, ls);
    else
        propose_prune(tree,     resid, pred_off, n, sigma2, cfg, rng, prob_prune, leaf_idx, leaf_counts, lps, (int)ls.size());
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

void sample_leaves(Tree& tree, const float* resid, const float* pred_off,
                   int n, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const std::vector<int>& leaf_idx, Workspace& ws) {
    float tau = cfg.leaf_prior_var;
    int   sz  = tree.full_size + 1;  // 128 for depth=6

    // Zero only the live portion of each lane (sz entries = 512 bytes each)
    std::memset(ws.s0, 0, sz * sizeof(float)); std::memset(ws.s1, 0, sz * sizeof(float));
    std::memset(ws.s2, 0, sz * sizeof(float)); std::memset(ws.s3, 0, sz * sizeof(float));
    std::memset(ws.c0, 0, sz * sizeof(int));   std::memset(ws.c1, 0, sz * sizeof(int));
    std::memset(ws.c2, 0, sz * sizeof(int));   std::memset(ws.c3, 0, sz * sizeof(int));

    int n4 = (n / 4) * 4;
    for (int i = 0; i < n4; i += 4) {
        int l0 = leaf_idx[i],   l1 = leaf_idx[i+1],
            l2 = leaf_idx[i+2], l3 = leaf_idx[i+3];
        ws.s0[l0] += resid[i]   + pred_off[i];   ws.c0[l0]++;
        ws.s1[l1] += resid[i+1] + pred_off[i+1]; ws.c1[l1]++;
        ws.s2[l2] += resid[i+2] + pred_off[i+2]; ws.c2[l2]++;
        ws.s3[l3] += resid[i+3] + pred_off[i+3]; ws.c3[l3]++;
    }
    for (int i = n4; i < n; i++) {
        int k = leaf_idx[i];
        ws.s0[k] += resid[i] + pred_off[i]; ws.c0[k]++;
    }

    for (int k = 1; k < sz; k++) {
        int   cnt = ws.c0[k] + ws.c1[k] + ws.c2[k] + ws.c3[k];
        if (cnt == 0) continue;
        float sum = ws.s0[k] + ws.s1[k] + ws.s2[k] + ws.s3[k];
        float post_mean = (tau * sum) / (cnt * tau + sigma2);
        float post_var  = (tau * sigma2) / (cnt * tau + sigma2);
        tree.leaf_value[k] = post_mean + std::sqrt(post_var) * rng.normal();
    }
}

void init_state(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees;
    state.trees.clear();
    for (int t = 0; t < T; t++) state.trees.emplace_back(cfg.tree_depth);
    state.pred.assign(T, std::vector<float>(state.n, 0.f));
    state.leaf_indices.assign(T, std::vector<int>(state.n, 1));  // all at root (node 1)
    int sz = state.trees[0].full_size + 1;
    state.leaf_counts.assign(T, std::vector<int>(sz, 0));
    for (int t = 0; t < T; t++) state.leaf_counts[t][1] = state.n;  // all obs at root
    state.residual.assign(state.y, state.y + state.n);
    state.sigma2 = 1.f;
    // Pre-allocate workspace scratch buffers
    state.ws.leaves_buf.reserve(state.trees[0].full_size + 1);
    state.ws.leaf_parents_buf.reserve(state.trees[0].half_size + 1);
    state.ws.zeros.assign(state.n, 0.f);
    (void)rng;
}

void mcmc_sweep(BARTState& s, const BARTConfig& cfg, RNG& rng) {
    for (int t = 0; t < cfg.num_trees; t++) {
        // v12: leaf_counts[t] eliminates O(n*leaves) grow_possible scan in propose_move
        propose_move(s.trees[t], s.Xq, s.residual.data(), s.pred[t].data(),
                     s.n, s.sigma2, cfg, rng, s.leaf_indices[t], s.leaf_counts[t], s.ws);

        sample_leaves(s.trees[t], s.residual.data(), s.pred[t].data(),
                      s.n, s.sigma2, cfg, rng, s.leaf_indices[t], s.ws);

        // Fused delta update: one pass instead of pred_update + subtract
        for (int i = 0; i < s.n; i++) {
            float pred_new   = s.trees[t].leaf_value[s.leaf_indices[t][i]];
            s.residual[i]   += s.pred[t][i] - pred_new;
            s.pred[t][i]     = pred_new;
        }
    }
    sample_sigma2(s.residual.data(), s.n, s.sigma2, cfg, rng);
}

} // namespace bart
