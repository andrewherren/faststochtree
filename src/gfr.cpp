#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
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
// eval_node — evaluate one node's split decision.
// Called both from the single-threaded path and the parallel path (gfr-v6).
// All inputs are read-only on the shared partition; outputs go to NodeDecision.
// -----------------------------------------------------------------------

struct NodeDecision {
    int     node_k;
    int     feat;      // -1 = no split
    uint8_t thresh;
    int     count_L;
};

static NodeDecision eval_node(int node_k, const TreePartition& part,
                               const QuantizedX& Xq, const float* resid,
                               int n, int p, int m, int n_k, float sum_T,
                               float sigma2, const BARTConfig& cfg,
                               RNG& local_rng, int depth) {
    struct FCand { uint8_t thresh; float sum_L; int count_L; };

    // Fisher-Yates: per-node scratch (small, stack-friendly)
    std::vector<int>   feat_order(p);
    std::iota(feat_order.begin(), feat_order.end(), 0);
    for (int k = 0; k < m; k++) {
        int swap = k + local_rng.randint(0, p - k);
        std::swap(feat_order[k], feat_order[swap]);
    }

    std::vector<float> feat_log_total(m + 1, -std::numeric_limits<float>::infinity());
    std::vector<float> resid_buf(n_k);
    int n_valid_feats = 0;

    float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
    p_split = std::min(p_split, 1.f - 1e-6f);
    float log_split_ratio = std::log(p_split) - std::log(1.f - p_split);
    int count_T = n_k;

    // Pass 1: per-feature log-sum-exp totals (online LSE, no storage)
    for (int fi = 0; fi < m; fi++) {
        int j = feat_order[fi];
        auto [beg_j, end_j] = part.ranges[j].at(node_k);
        const auto& w = part.working[j];
        for (int k = beg_j; k < end_j; k++)
            resid_buf[k - beg_j] = resid[w[k]];
        float sum_L = 0.f; int count_L = 0;
        float max_lw = -std::numeric_limits<float>::infinity(), sum_exp = 0.f;
        for (int k = 0; k < n_k - 1; k++) {
            sum_L += resid_buf[k]; count_L++;
            int obs_i = w[beg_j + k];
            if (Xq.at(obs_i, j) >= Xq.at(w[beg_j + k + 1], j)) continue;
            int count_R = count_T - count_L;
            if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
            float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                         + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
            if (log_ml > max_lw) { sum_exp = sum_exp * std::exp(max_lw - log_ml) + 1.f; max_lw = log_ml; }
            else sum_exp += std::exp(log_ml - max_lw);
        }
        if (sum_exp > 0.f) { feat_log_total[fi] = max_lw + std::log(sum_exp); n_valid_feats++; }
    }
    feat_log_total[m] = leaf_log_ml(sum_T, count_T, sigma2, cfg.leaf_prior_var)
                      - log_split_ratio
                      + (n_valid_feats > 0 ? std::log((float)n_valid_feats) : 0.f);

    // Stage 1: sample feature
    float max_ft = *std::max_element(feat_log_total.begin(), feat_log_total.end());
    float total_ft = 0.f;
    for (int fi = 0; fi <= m; fi++) total_ft += std::exp(feat_log_total[fi] - max_ft);
    float u1 = local_rng.uniform() * total_ft, cum1 = 0.f;
    int chosen_fi = m;
    for (int fi = 0; fi < m; fi++) {
        cum1 += std::exp(feat_log_total[fi] - max_ft);
        if (u1 <= cum1) { chosen_fi = fi; break; }
    }
    if (chosen_fi == m) return {node_k, -1, 0, 0};  // no split
    int chosen_feat = feat_order[chosen_fi];

    // Pass 2: re-scan chosen feature, collect candidates for stage-2
    std::vector<FCand>  cands;
    std::vector<float>  log_wts;
    {
        auto [beg_j, end_j] = part.ranges[chosen_feat].at(node_k);
        const auto& w = part.working[chosen_feat];
        for (int k = beg_j; k < end_j; k++) resid_buf[k - beg_j] = resid[w[k]];
        float sum_L = 0.f; int count_L = 0;
        for (int k = 0; k < n_k - 1; k++) {
            sum_L += resid_buf[k]; count_L++;
            int obs_i = w[beg_j + k];
            if (Xq.at(obs_i, chosen_feat) >= Xq.at(w[beg_j + k + 1], chosen_feat)) continue;
            int count_R = count_T - count_L;
            if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
            float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                         + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
            cands.push_back({Xq.at(obs_i, chosen_feat), sum_L, count_L});
            log_wts.push_back(log_ml);
        }
    }

    // Stage 2: softmax over cutpoints
    float max_lw2 = *std::max_element(log_wts.begin(), log_wts.end());
    float total_ct = 0.f;
    for (float lw : log_wts) total_ct += std::exp(lw - max_lw2);
    float u2 = local_rng.uniform() * total_ct, cum2 = 0.f;
    int chosen_cut = (int)cands.size() - 1;
    for (int k = 0; k < (int)cands.size() - 1; k++) {
        cum2 += std::exp(log_wts[k] - max_lw2);
        if (u2 <= cum2) { chosen_cut = k; break; }
    }
    return {node_k, chosen_feat, cands[chosen_cut].thresh, cands[chosen_cut].count_L};
}

// -----------------------------------------------------------------------
// grow_tree_gfr (v6) — level-BFS with optional parallel node evaluation
//
// Nodes at the same BFS level have disjoint observation sets → independent.
// Evaluation (histogram scan + sampling) runs in parallel via thread pool.
// tree.grow + update_partition applied sequentially after collecting results.
//
// pool=nullptr → single-threaded (same as v5 but with level-BFS structure).
// -----------------------------------------------------------------------

void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps, ThreadPool* pool) {
    tree.reset();

    TreePartition part;
    part.init(ps, n, p);

    int m = (cfg.p_eval > 0 && cfg.p_eval < p) ? cfg.p_eval : p;

    std::vector<int>          current_level = {1};
    std::vector<NodeDecision> decisions;

    while (!current_level.empty()) {
        // Filter: drop nodes below size/depth thresholds
        std::vector<int>   active;
        std::vector<int>   node_nk;
        std::vector<float> node_sumT;
        std::vector<int>   node_depth;

        for (int node_k : current_level) {
            auto it = part.ranges[0].find(node_k);
            if (it == part.ranges[0].end()) continue;
            auto [beg0, end0] = it->second;
            int n_k = end0 - beg0;
            int depth = Tree::depth_of(node_k);
            if (n_k < 2 * cfg.min_samples_leaf || depth >= tree.depth - 1) {
                for (int j = 0; j < p; j++) part.ranges[j].erase(node_k);
                continue;
            }
            float sum_T = 0.f;
            const auto& w0 = part.working[0];
            for (int k = beg0; k < end0; k++) sum_T += resid[w0[k]];
            active.push_back(node_k);
            node_nk.push_back(n_k);
            node_sumT.push_back(sum_T);
            node_depth.push_back(depth);
        }
        if (active.empty()) break;

        // Pre-generate per-node RNG seeds sequentially (preserves global RNG stream)
        std::vector<unsigned> seeds(active.size());
        for (auto& s : seeds) s = (unsigned)rng.randint(0, INT_MAX);

        decisions.resize(active.size());

        // Evaluate nodes — parallel if pool provided and work is large enough
        bool do_parallel = pool && ((int)active.size() * node_nk[0] * m > 500'000);

        auto eval = [&](int ki) {
            RNG local_rng(seeds[ki]);
            decisions[ki] = eval_node(active[ki], part, Xq, resid, n, p, m,
                                       node_nk[ki], node_sumT[ki], sigma2, cfg,
                                       local_rng, node_depth[ki]);
        };

        if (do_parallel) pool->parallel_for(0, (int)active.size(), eval);
        else             for (int ki = 0; ki < (int)active.size(); ki++) eval(ki);

        // Apply splits sequentially (tree + partition writes are not thread-safe)
        std::vector<int> next_level;
        for (const auto& d : decisions) {
            if (d.feat == -1) continue;
            tree.grow(d.node_k, d.feat, d.thresh);
            update_partition(part, d.node_k, d.feat, d.thresh, d.count_L, Xq, p);
            next_level.push_back(2 * d.node_k);
            next_level.push_back(2 * d.node_k + 1);
        }
        current_level = std::move(next_level);
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
                      n, state.p, state.sigma2, cfg, rng, state.presorted,
                      state.thread_pool.get());

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
