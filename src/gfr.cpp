#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include <algorithm>
#include <cmath>
#include <deque>
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
// grow_tree_gfr (v3+v4 revised) — true two-pass two-stage sampling
//
// Pass 1: stream all p features, compute per-feature log-sum-exp totals
//         via online LSE.  No candidate storage — just p+1 floats.
// Stage 1: softmax over p+1 totals, select feature (or no-split).
// Pass 2: re-scan chosen feature only to collect candidates for stage-2.
//
// Memory per node: O(p) for totals + O(n_k) for chosen feature's candidates.
// Also uses a gather buffer (size n) to enable sequential prefix-sum reads.
// -----------------------------------------------------------------------

void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   const PresortedX& ps) {
    tree.reset();

    TreePartition part;
    part.init(ps, n, p);

    struct FCand { uint8_t thresh; float sum_L; int count_L; };
    std::vector<FCand>  cands_buf;    // pass-2 candidates for chosen feature
    std::vector<float>  log_wts_buf;  // pass-2 log-weights for chosen feature
    std::vector<float>  feat_log_total(p + 1);
    std::vector<float>  resid_buf(n);

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

        float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
        p_split = std::min(p_split, 1.f - 1e-6f);
        float log_split_ratio = std::log(p_split) - std::log(1.f - p_split);

        // Pass 1: per-feature log-sum-exp totals via online LSE — no candidate storage
        std::fill(feat_log_total.begin(), feat_log_total.end(),
                  -std::numeric_limits<float>::infinity());
        int n_valid_feats = 0;

        for (int j = 0; j < p; j++) {
            auto [beg_j, end_j] = part.ranges[j].at(node_k);
            const auto& w = part.working[j];

            // Gather: scattered resid[w[k]] → contiguous resid_buf — enables vectorised prefix sum
            for (int k = beg_j; k < end_j; k++)
                resid_buf[k - beg_j] = resid[w[k]];

            // Prefix sum + online LSE (single pass, no candidate storage)
            float sum_L = 0.f; int count_L = 0;
            float max_lw = -std::numeric_limits<float>::infinity(), sum_exp = 0.f;

            for (int k = 0; k < n_k - 1; k++) {
                sum_L += resid_buf[k]; count_L++;
                int obs_i = w[beg_j + k];
                if (Xq.at(obs_i, j) >= Xq.at(w[beg_j + k + 1], j)) continue;  // tie
                int count_R = count_T - count_L;
                if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                             + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                // Online LSE: update running max and shifted sum
                if (log_ml > max_lw) {
                    sum_exp = sum_exp * std::exp(max_lw - log_ml) + 1.f;
                    max_lw = log_ml;
                } else {
                    sum_exp += std::exp(log_ml - max_lw);
                }
            }
            if (sum_exp > 0.f) {
                feat_log_total[j] = max_lw + std::log(sum_exp);
                n_valid_feats++;
            }
        }

        // No-split weight at slot p
        feat_log_total[p] = leaf_log_ml(sum_T, count_T, sigma2, cfg.leaf_prior_var)
                          - log_split_ratio
                          + (n_valid_feats > 0 ? std::log((float)n_valid_feats) : 0.f);

        // Stage 1: sample feature from p+1 totals (on-the-fly exp, no feat_wts vector)
        float max_ft = *std::max_element(feat_log_total.begin(), feat_log_total.end());
        float total_ft = 0.f;
        for (int j = 0; j <= p; j++) total_ft += std::exp(feat_log_total[j] - max_ft);
        float u1 = rng.uniform() * total_ft, cum1 = 0.f;
        int chosen_feat = p;
        for (int j = 0; j < p; j++) {
            cum1 += std::exp(feat_log_total[j] - max_ft);
            if (u1 <= cum1) { chosen_feat = j; break; }
        }

        if (chosen_feat == p) {
            for (int j = 0; j < p; j++) part.ranges[j].erase(node_k);
            continue;
        }

        // Pass 2: re-scan chosen feature only — collect candidates for stage-2
        cands_buf.clear();
        log_wts_buf.clear();
        {
            auto [beg_j, end_j] = part.ranges[chosen_feat].at(node_k);
            const auto& w = part.working[chosen_feat];
            for (int k = beg_j; k < end_j; k++)
                resid_buf[k - beg_j] = resid[w[k]];
            float sum_L = 0.f; int count_L = 0;
            for (int k = 0; k < n_k - 1; k++) {
                sum_L += resid_buf[k]; count_L++;
                int obs_i = w[beg_j + k];
                if (Xq.at(obs_i, chosen_feat) >= Xq.at(w[beg_j + k + 1], chosen_feat)) continue;
                int count_R = count_T - count_L;
                if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                             + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                cands_buf.push_back({Xq.at(obs_i, chosen_feat), sum_L, count_L});
                log_wts_buf.push_back(log_ml);
            }
        }

        // Stage 2: softmax over cutpoints of chosen feature (on-the-fly exp)
        float max_lw2 = *std::max_element(log_wts_buf.begin(), log_wts_buf.end());
        float total_ct = 0.f;
        for (float lw : log_wts_buf) total_ct += std::exp(lw - max_lw2);
        float u2 = rng.uniform() * total_ct, cum2 = 0.f;
        int chosen_cut = (int)cands_buf.size() - 1;
        for (int k = 0; k < (int)cands_buf.size() - 1; k++) {
            cum2 += std::exp(log_wts_buf[k] - max_lw2);
            if (u2 <= cum2) { chosen_cut = k; break; }
        }

        const auto& c = cands_buf[chosen_cut];
        tree.grow(node_k, chosen_feat, c.thresh);
        update_partition(part, node_k, chosen_feat, c.thresh, c.count_L, Xq, p);
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
