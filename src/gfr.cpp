#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace bart {

// -----------------------------------------------------------------------
// PresortedX::build — argsort each feature once; reused across all tree
// rebuilds in a run.  O(n*p*log n) total, amortized to O(1) per rebuild.
// -----------------------------------------------------------------------

void PresortedX::build(const QuantizedX& Xq, int n, int p) {
    sorted_idx.resize(p);
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    for (int j = 0; j < p; j++) {
        sorted_idx[j] = idx;
        std::sort(sorted_idx[j].begin(), sorted_idx[j].end(),
                  [&](int a, int b) { return Xq.at(a, j) < Xq.at(b, j); });
    }
}

// -----------------------------------------------------------------------
// TreePartition — per-tree mutable copy of sorted_idx with [beg,end)
// range tracking per node per feature.
// -----------------------------------------------------------------------

struct TreePartition {
    std::vector<std::vector<int>>                          working;  // [p][n]
    std::vector<std::unordered_map<int,std::pair<int,int>>> ranges;  // [p][node→{beg,end}]

    void init(const PresortedX& ps, int n, int p) {
        working = ps.sorted_idx;  // deep copy; O(n*p) — eliminated in gfr-v8
        ranges.assign(p, {});
        for (int j = 0; j < p; j++) ranges[j][1] = {0, n};
    }
};

// Stable-partition all p feature working arrays after a split at node_k.
// For the split feature, the pre-sorted order already separates left/right.
// For all other features, we partition in-place preserving sorted order.
static void update_partition(TreePartition& part, int node_k,
                              int feat, uint8_t thresh, int count_L,
                              const QuantizedX& Xq, int p) {
    std::vector<int> right_buf;
    for (int j = 0; j < p; j++) {
        auto [beg, end] = part.ranges[j].at(node_k);
        auto& w = part.working[j];
        if (j == feat) {
            // Split feature: sorted array is already left|right at beg+count_L
            part.ranges[j][2*node_k]   = {beg, beg + count_L};
            part.ranges[j][2*node_k+1] = {beg + count_L, end};
        } else {
            // Other features: stable-partition by split criterion
            right_buf.clear();
            int write = beg;
            for (int k = beg; k < end; k++) {
                int obs = w[k];
                if (Xq.at(obs, feat) <= thresh) w[write++] = obs;
                else right_buf.push_back(obs);
            }
            int left_count = write - beg;
            for (int obs : right_buf) w[write++] = obs;
            part.ranges[j][2*node_k]   = {beg, beg + left_count};
            part.ranges[j][2*node_k+1] = {beg + left_count, end};
        }
        part.ranges[j].erase(node_k);
    }
}

// -----------------------------------------------------------------------
// grow_tree_gfr (v2) — O(n*p) per rebuild using presorted indices
// -----------------------------------------------------------------------

void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps) {
    tree.reset();

    TreePartition part;
    part.init(ps, n, p);

    std::deque<int> queue;
    queue.push_back(1);

    while (!queue.empty()) {
        int node_k = queue.front(); queue.pop_front();

        auto it = part.ranges[0].find(node_k);
        if (it == part.ranges[0].end()) continue;
        auto [beg0, end0] = it->second;
        int n_k = end0 - beg0;

        int depth = Tree::depth_of(node_k);
        if (n_k < 2 * cfg.min_samples_leaf || depth >= tree.depth - 1) {
            for (int j = 0; j < p; j++) part.ranges[j].erase(node_k);
            continue;
        }

        // Total sufficient stats from feature-0's sorted range
        float sum_T = 0.f;
        const auto& w0 = part.working[0];
        for (int k = beg0; k < end0; k++) sum_T += resid[w0[k]];
        int count_T = n_k;

        // Build flat candidate list — one O(n_k) pass per feature
        struct Cand { int feat; uint8_t thresh; float sum_L; int count_L; };
        std::vector<Cand>  cands;
        std::vector<float> log_wts;

        for (int j = 0; j < p; j++) {
            auto [beg_j, end_j] = part.ranges[j].at(node_k);
            const auto& w = part.working[j];
            float sum_L = 0.f; int count_L = 0;
            for (int k = beg_j; k < end_j - 1; k++) {
                int obs_i = w[k];
                sum_L += resid[obs_i]; count_L++;
                if (Xq.at(obs_i, j) >= Xq.at(w[k+1], j)) continue;  // tie
                int count_R = count_T - count_L;
                if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                             + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                cands.push_back({j, Xq.at(obs_i, j), sum_L, count_L});
                log_wts.push_back(log_ml);
            }
        }

        // No-split option
        int   n_valid = (int)cands.size();
        float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
        p_split = std::min(p_split, 1.f - 1e-6f);
        float no_wt = leaf_log_ml(sum_T, count_T, sigma2, cfg.leaf_prior_var)
                    + std::log(1.f - p_split) - std::log(p_split)
                    + (n_valid > 0 ? std::log((float)n_valid) : 0.f);
        cands.push_back({-1, 0, 0.f, 0});
        log_wts.push_back(no_wt);

        // Softmax sample
        float max_lw = *std::max_element(log_wts.begin(), log_wts.end());
        std::vector<float> wts(log_wts.size());
        float total = 0.f;
        for (int k = 0; k < (int)log_wts.size(); k++) {
            wts[k] = std::exp(log_wts[k] - max_lw); total += wts[k];
        }
        float u = rng.uniform() * total, cum = 0.f;
        int chosen = (int)wts.size() - 1;
        for (int k = 0; k < (int)wts.size() - 1; k++) {
            cum += wts[k];
            if (u <= cum) { chosen = k; break; }
        }

        const auto& c = cands[chosen];
        if (c.feat == -1) {
            for (int j = 0; j < p; j++) part.ranges[j].erase(node_k);
            continue;
        }

        tree.grow(node_k, c.feat, c.thresh);
        update_partition(part, node_k, c.feat, c.thresh, c.count_L, Xq, p);
        queue.push_front(2 * node_k + 1);
        queue.push_front(2 * node_k);
    }
}

// -----------------------------------------------------------------------
// gfr_sweep
// -----------------------------------------------------------------------

void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees, n = state.n;
    const uint8_t* xq = state.Xq.data.data();

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) state.residual[i] += state.pred[t][i];

        grow_tree_gfr(state.trees[t], state.Xq, state.residual.data(),
                      n, state.p, state.sigma2, cfg, rng, state.presorted);

        // Rebuild leaf index cache — tree was rebuilt from scratch
        for (int i = 0; i < n; i++)
            state.leaf_indices[t][i] = state.trees[t].traverse(xq, i, n);

        // zeros as pred_off: residual already fully restored above
        sample_leaves(state.trees[t], state.residual.data(), state.ws.zeros.data(),
                      n, state.sigma2, cfg, rng, state.leaf_indices[t], state.ws);

        for (int i = 0; i < n; i++)
            state.pred[t][i] = state.trees[t].leaf_value[state.leaf_indices[t][i]];
        for (int i = 0; i < n; i++) state.residual[i] -= state.pred[t][i];
    }
    sample_sigma2(state.residual.data(), n, state.sigma2, cfg, rng);
}

} // namespace bart
