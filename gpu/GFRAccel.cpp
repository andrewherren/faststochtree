#include "GFRAccel.h"
#include "faststochtree/model.hpp"
#include "faststochtree/mcmc.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace {

static void rebuild_leaf_state(bart::BARTState& s, int t, int n) {
    auto& ws = s.gfr_hist_ws;
    std::copy(ws.flat_obs.begin(), ws.flat_obs.begin() + n, s.flat_obs[t].begin());
    std::fill(s.leaf_counts[t].begin(), s.leaf_counts[t].end(), 0);
    for (auto& seg : ws.leaf_segs) {
        s.leaf_counts[t][seg.node] = seg.end - seg.beg;
        for (int idx = seg.beg; idx < seg.end; idx++)
            s.leaf_indices[t][s.flat_obs[t][idx]] = seg.node;
    }
    std::fill(s.leaf_start[t].begin(), s.leaf_start[t].end(), 0);
    for (auto& seg : ws.leaf_segs) s.leaf_start[t][seg.node] = seg.beg;
}

} // namespace

namespace gpu {

void grow_tree_gfr_gpu(bart::Tree& tree, const bart::QuantizedX& Xq,
                       const float* resid, int n, int p, float sigma2,
                       const bart::BARTConfig& cfg, bart::RNG& rng,
                       bart::GFRHistWorkspace& ws, MetalContext& gpu_ctx)
{
    tree.reset();
    ws.reinit(n);
    const int m = p;  // no feature subsampling (p_eval == 0 required)

    const float NEG_INF = -std::numeric_limits<float>::infinity();

    std::vector<float> gpu_sum;
    std::vector<int>   gpu_cnt;
    std::vector<float> gpu_feat_log;

    std::vector<int> current_level = {1};

    while (!current_level.empty()) {
        std::vector<int> next_level;

        // Collect nodes eligible to split at this level.
        struct ActiveNode { int node_k, beg, end; };
        std::vector<ActiveNode> active;
        active.reserve(current_level.size());

        for (int node_k : current_level) {
            auto& nr = ws.node_range[node_k];
            if (nr.first == -1) continue;
            int beg = nr.first, end = nr.second;
            int n_k   = end - beg;
            int depth = bart::Tree::depth_of(node_k);
            if (n_k < 2 * cfg.min_samples_leaf || depth >= tree.depth - 1) {
                ws.leaf_segs.push_back({node_k, beg, end});
                nr.first = -1;
                continue;
            }
            active.push_back({node_k, beg, end});
        }

        if (active.empty()) {
            current_level = std::move(next_level);
            continue;
        }

        // One GPU command buffer for all active nodes at this level.
        const int n_active = (int)active.size();
        std::vector<int> node_ranges_buf(n_active * 2);
        for (int ai = 0; ai < n_active; ai++) {
            node_ranges_buf[2*ai]   = active[ai].beg;
            node_ranges_buf[2*ai+1] = active[ai].end;
        }
        gpu_sum.resize((size_t)n_active * p * 256);
        gpu_cnt.resize((size_t)n_active * p * 256);
        gpu_feat_log.resize((size_t)n_active * p);

        // Two-pass dispatch: histogram then scan in one MTLCommandBuffer.
        // obs_count = n so flat_obs ranges index into the full array absolutely.
        gpu_ctx.histogram_and_scan(
            Xq.data.data(), resid,
            ws.flat_obs.data(), n,
            node_ranges_buf.data(), n, p, n_active,
            sigma2, cfg.leaf_prior_var, cfg.min_samples_leaf,
            gpu_sum.data(), gpu_cnt.data(), gpu_feat_log.data());

        // Per-node: CPU scan over GPU histograms + stage1/stage2/partition.
        for (int ai = 0; ai < n_active; ai++) {
            int node_k = active[ai].node_k;
            int beg    = active[ai].beg;
            int end    = active[ai].end;
            int depth  = bart::Tree::depth_of(node_k);

            const float* node_sum = gpu_sum.data() + (size_t)ai * p * 256;
            const int*   node_cnt = gpu_cnt.data() + (size_t)ai * p * 256;

            // Derive sum_T / count_T from feature 0's histogram (Σ bins = node total).
            float sum_T = 0.f; int count_T = 0;
            for (int b = 0; b < 256; b++) { sum_T += node_sum[b]; count_T += node_cnt[b]; }

            bart::RNG local_rng(rng.randint(0, INT_MAX));

            float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
            p_split = std::min(p_split, 1.f - 1e-6f);
            float log_split_ratio = std::log(p_split) - std::log(1.f - p_split);

            // feat_log_total[0..m-1] already computed by the GPU scan kernel.
            const float* node_feat_log = gpu_feat_log.data() + ai * p;
            for (int fi = 0; fi < m; fi++)
                ws.feat_log_total[fi] = node_feat_log[fi];

            // Stage 1: softmax over features + no-split option.
            int n_valid_feats = 0;
            for (int fi = 0; fi < m; fi++)
                if (ws.feat_log_total[fi] > NEG_INF) n_valid_feats++;

            ws.feat_log_total[m] =
                bart::leaf_log_ml(sum_T, (float)count_T, sigma2, cfg.leaf_prior_var)
                - log_split_ratio
                + (n_valid_feats > 0 ? std::log((float)n_valid_feats) : 0.f);

            float max_ft = *std::max_element(ws.feat_log_total.data(), ws.feat_log_total.data() + m + 1);
            float total_ft = 0.f;
            for (int fi = 0; fi <= m; fi++) total_ft += std::exp(ws.feat_log_total[fi] - max_ft);
            float u1 = local_rng.uniform() * total_ft, cum1 = 0.f;
            int chosen_fi = m;
            for (int fi = 0; fi < m; fi++) {
                cum1 += std::exp(ws.feat_log_total[fi] - max_ft);
                if (u1 <= cum1) { chosen_fi = fi; break; }
            }
            if (chosen_fi == m) {
                ws.leaf_segs.push_back({node_k, beg, end});
                ws.node_range[node_k].first = -1;
                continue;
            }
            int chosen_feat = chosen_fi;  // identity feat_order: fi == feature index

            // Stage 2: cutpoint softmax over chosen feature's GPU histogram.
            ws.cut_log_wts.clear();
            ws.cut_thresh_buf.clear();
            {
                const float* sh = node_sum + chosen_fi * 256;
                const int*   ch = node_cnt + chosen_fi * 256;
                float sum_L = 0.f; int count_L = 0;
                for (int b = 0; b < 255; b++) {
                    if (ch[b] == 0) continue;
                    sum_L   += sh[b]; count_L += ch[b];
                    int count_R = count_T - count_L;
                    if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                    float log_ml = bart::leaf_log_ml(sum_L,           (float)count_L, sigma2, cfg.leaf_prior_var)
                                 + bart::leaf_log_ml(sum_T - sum_L,   (float)count_R, sigma2, cfg.leaf_prior_var);
                    ws.cut_log_wts.push_back(log_ml);
                    ws.cut_thresh_buf.push_back((uint8_t)b);
                }
            }
            if (ws.cut_thresh_buf.empty()) {
                ws.leaf_segs.push_back({node_k, beg, end});
                ws.node_range[node_k].first = -1;
                continue;
            }

            float max_lw2 = *std::max_element(ws.cut_log_wts.begin(), ws.cut_log_wts.end());
            float total_ct = 0.f;
            for (float lw : ws.cut_log_wts) total_ct += std::exp(lw - max_lw2);
            float u2 = local_rng.uniform() * total_ct, cum2 = 0.f;
            int chosen_cut = (int)ws.cut_thresh_buf.size() - 1;
            for (int k = 0; k < (int)ws.cut_thresh_buf.size() - 1; k++) {
                cum2 += std::exp(ws.cut_log_wts[k] - max_lw2);
                if (u2 <= cum2) { chosen_cut = k; break; }
            }
            uint8_t thresh = ws.cut_thresh_buf[chosen_cut];

            tree.grow(node_k, chosen_feat, thresh);

            // Scalar partition (compress8 is in gfr.cpp anonymous namespace).
            const uint8_t* col = Xq.data.data() + chosen_feat * Xq.n;
            ws.right_buf.clear();
            int write = beg;
            for (int k = beg; k < end; k++) {
                int ob = ws.flat_obs[k];
                if (col[ob] <= thresh) ws.flat_obs[write++] = ob;
                else                   ws.right_buf.push_back(ob);
            }
            int left_end = write;
            for (int ob : ws.right_buf) ws.flat_obs[write++] = ob;

            ws.node_range[node_k].first = -1;
            if (left_end > beg)   ws.node_range[2 * node_k]     = {beg, left_end};
            if (write > left_end) ws.node_range[2 * node_k + 1] = {left_end, end};

            next_level.push_back(2 * node_k);
            next_level.push_back(2 * node_k + 1);
        }

        current_level = std::move(next_level);
    }
}

void gfr_sweep_gpu(bart::BARTState& state, const bart::BARTConfig& cfg,
                   bart::RNG& rng, MetalContext& gpu_ctx)
{
    int T = cfg.num_trees, n = state.n;

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) state.residual[i] += state.pred[t][i];

        grow_tree_gfr_gpu(state.trees[t], state.Xq, state.residual.data(),
                          n, state.p, state.sigma2, cfg, rng,
                          state.gfr_hist_ws, gpu_ctx);

        rebuild_leaf_state(state, t, n);

        bart::sample_leaves(state.trees[t], state.residual.data(), state.ws.zeros.data(),
                            n, state.sigma2, cfg, rng, state.leaf_indices[t], state.ws);

        for (int i = 0; i < n; i++)
            state.pred[t][i] = state.trees[t].leaf_value[state.leaf_indices[t][i]];
        for (int i = 0; i < n; i++) state.residual[i] -= state.pred[t][i];
    }
    bart::sample_sigma2(state.residual.data(), n, state.sigma2, cfg, rng);
}

} // namespace gpu
