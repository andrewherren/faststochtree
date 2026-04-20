#include "faststochtree/gfr.hpp"
#include "faststochtree/mcmc.hpp"
#include <algorithm>
#include <cmath>
#include <climits>
#include <limits>
#include <numeric>
#include <vector>

// MSVC doesn't support __builtin_prefetch; make it a no-op there.
#ifndef __GNUC__
#  define __builtin_prefetch(addr, ...) ((void)(addr))
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
// compress8: lookup table for branchless 8-lane compress (NEON partition).
// compress8.idx[mask] — source indices permuted so set-bit lanes (left obs)
//   come first, then clear-bit lanes (right obs).
// compress8.cnt[mask] — number of set bits (obs going left).
// 256 entries × 8 bytes = 2 KB; fits comfortably in L1.
namespace {
struct Compress8Table {
    uint8_t idx[256][8];
    int     cnt[256];
    Compress8Table() noexcept {
        for (int mask = 0; mask < 256; mask++) {
            int nl = 0, nr = 0;
            uint8_t left[8], right[8];
            for (int i = 0; i < 8; i++) {
                if (mask & (1 << i)) left[nl++]  = (uint8_t)i;
                else                 right[nr++] = (uint8_t)i;
            }
            cnt[mask] = nl;
            for (int i = 0; i < nl; i++) idx[mask][i]      = left[i];
            for (int i = 0; i < nr; i++) idx[mask][nl + i] = right[i];
        }
    }
};
static const Compress8Table compress8;
} // namespace
#endif // __ARM_NEON

namespace bart {

// -----------------------------------------------------------------------
// grow_tree_gfr (v12) — NEON compress partition + prefetch histogram
//
// Key design changes vs v9:
//   5. node_range: unordered_map<int,pair<int,int>> replaced by flat
//      vector<pair<int,int>> indexed directly by node id (bounded by
//      full_size = 2^(depth+1)-1).  Eliminates hash overhead on every
//      node lookup, insert, and erase.  Sentinel beg==-1 marks inactive.
//
// Retained from v9:
//   1. Single flat_obs list partitioned in-place — O(n_k), not O(n_k*p).
//   2. 256-bin sum/count histograms per feature — O(n_k*m) build,
//      O(256*m) prefix scan.
//   3. Fused stage-1/stage-2: one histogram pass, O(256) stage-2 rescan.
//   4. Persistent GFRHistWorkspace — no per-call heap allocation.
//
// Parallelism: pool->parallel_for(0, m, hist_build) when m*n_k > 200,000.
//   Each feature fi owns ws.sum_hists[fi*256..] and ws.cnt_hists[fi*256..]
//   → no data race.  Stage 1/2 softmax and obs-list partition are sequential.
// -----------------------------------------------------------------------

void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng,
                   GFRHistWorkspace& ws, ThreadPool* pool) {
    tree.reset();
    ws.reinit(n);

    int m = (cfg.p_eval > 0 && cfg.p_eval < p) ? cfg.p_eval : p;

    // feat_order scratch — reused each node via Fisher-Yates
    ws.feat_order.resize(p);
    std::iota(ws.feat_order.begin(), ws.feat_order.end(), 0);

    std::vector<int> current_level = {1};

    while (!current_level.empty()) {
        std::vector<int> next_level;

        for (int node_k : current_level) {
            auto& nr = ws.node_range[node_k];
            if (nr.first == -1) continue;
            int beg = nr.first, end = nr.second;
            int n_k = end - beg;
            int depth = Tree::depth_of(node_k);

            if (n_k < 2 * cfg.min_samples_leaf || depth >= tree.depth - 1) {
                ws.leaf_segs.push_back({node_k, beg, end});
                nr.first = -1;
                continue;
            }

            // Compute total residual sum for this node
            float sum_T = 0.f;
            for (int k = beg; k < end; k++) sum_T += resid[ws.flat_obs[k]];
            int count_T = n_k;

            // Fisher-Yates shuffle first m of feat_order
            // Restore canonical order first (reuse the same vector)
            std::iota(ws.feat_order.begin(), ws.feat_order.end(), 0);
            RNG local_rng(rng.randint(0, INT_MAX));
            for (int k = 0; k < m; k++) {
                int swap = k + local_rng.randint(0, p - k);
                std::swap(ws.feat_order[k], ws.feat_order[swap]);
            }

            float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
            p_split = std::min(p_split, 1.f - 1e-6f);
            float log_split_ratio = std::log(p_split) - std::log(1.f - p_split);

            // Initialise feat_log_total to -inf
            const float NEG_INF = -std::numeric_limits<float>::infinity();
            for (int fi = 0; fi <= m; fi++) ws.feat_log_total[fi] = NEG_INF;

            // ---------------------------------------------------------------
            // hist_build lambda: for feature fi build histogram then prefix-
            // scan to compute feat_log_total[fi].
            // Each fi owns a contiguous 256-element slice → no race.
            // ---------------------------------------------------------------
            auto hist_build = [&](int fi) {
                int j = ws.feat_order[fi];
                float* sh = ws.sum_hists.data() + fi * 256;
                int*   ch = ws.cnt_hists.data() + fi * 256;
                std::fill(sh, sh + 256, 0.f);
                std::fill(ch, ch + 256, 0);

                // Build histogram over obs in [beg, end)
                // Prefetch + 4x unroll to hide indirect-load latency.
                // Gate on n_k: below ~128 obs the overhead isn't worth it.
                constexpr int PF = 12;
                int hk = beg;
                if (n_k >= 128) {
                    const uint8_t* col = Xq.data.data() + j * Xq.n;
                    for (; hk + 3 < end; hk += 4) {
                        if (hk + PF + 3 < end) {
                            __builtin_prefetch(col + ws.flat_obs[hk + PF],     0, 1);
                            __builtin_prefetch(col + ws.flat_obs[hk + PF + 1], 0, 1);
                            __builtin_prefetch(col + ws.flat_obs[hk + PF + 2], 0, 1);
                            __builtin_prefetch(col + ws.flat_obs[hk + PF + 3], 0, 1);
                            __builtin_prefetch(resid + ws.flat_obs[hk + PF],     0, 1);
                            __builtin_prefetch(resid + ws.flat_obs[hk + PF + 1], 0, 1);
                            __builtin_prefetch(resid + ws.flat_obs[hk + PF + 2], 0, 1);
                            __builtin_prefetch(resid + ws.flat_obs[hk + PF + 3], 0, 1);
                        }
                        int o0 = ws.flat_obs[hk],   o1 = ws.flat_obs[hk+1];
                        int o2 = ws.flat_obs[hk+2], o3 = ws.flat_obs[hk+3];
                        uint8_t b0 = col[o0], b1 = col[o1];
                        uint8_t b2 = col[o2], b3 = col[o3];
                        sh[b0] += resid[o0]; ch[b0]++;
                        sh[b1] += resid[o1]; ch[b1]++;
                        sh[b2] += resid[o2]; ch[b2]++;
                        sh[b3] += resid[o3]; ch[b3]++;
                    }
                }
                for (; hk < end; hk++) {
                    int obs = ws.flat_obs[hk];
                    uint8_t bin = Xq.at(obs, j);
                    sh[bin] += resid[obs];
                    ch[bin]++;
                }

                // Prefix scan: online LSE over valid split points
                float sum_L = 0.f; int count_L = 0;
                float max_lw = NEG_INF, sum_exp = 0.f;
                for (int b = 0; b < 255; b++) {
                    if (ch[b] == 0) continue;
                    sum_L += sh[b]; count_L += ch[b];
                    int count_R = count_T - count_L;
                    if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                    float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                                 + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                    if (log_ml > max_lw) { sum_exp = sum_exp * std::exp(max_lw - log_ml) + 1.f; max_lw = log_ml; }
                    else sum_exp += std::exp(log_ml - max_lw);
                }
                if (sum_exp > 0.f) ws.feat_log_total[fi] = max_lw + std::log(sum_exp);
            };

            bool par_hist = pool && ((long)m * n_k > 200'000L);
            if (par_hist) pool->parallel_for(0, m, hist_build);
            else          for (int fi = 0; fi < m; fi++) hist_build(fi);

            // ---------------------------------------------------------------
            // Stage 1: softmax over features + no-split
            // ---------------------------------------------------------------
            int n_valid_feats = 0;
            for (int fi = 0; fi < m; fi++)
                if (ws.feat_log_total[fi] > NEG_INF) n_valid_feats++;

            ws.feat_log_total[m] = leaf_log_ml(sum_T, count_T, sigma2, cfg.leaf_prior_var)
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
                nr.first = -1;
                continue;  // no split
            }
            int chosen_feat = ws.feat_order[chosen_fi];

            // ---------------------------------------------------------------
            // Stage 2: softmax over cutpoints using histogram of chosen_feat
            // Histogram already built in hist_build — re-scan in O(256).
            // ---------------------------------------------------------------
            ws.cut_log_wts.clear();
            ws.cut_thresh_buf.clear();
            {
                float* sh = ws.sum_hists.data() + chosen_fi * 256;
                int*   ch = ws.cnt_hists.data() + chosen_fi * 256;
                float sum_L = 0.f; int count_L = 0;
                for (int b = 0; b < 255; b++) {
                    if (ch[b] == 0) continue;
                    sum_L += sh[b]; count_L += ch[b];
                    int count_R = count_T - count_L;
                    if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                    float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                                 + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                    ws.cut_log_wts.push_back(log_ml);
                    ws.cut_thresh_buf.push_back((uint8_t)b);
                }
            }

            if (ws.cut_thresh_buf.empty()) {
                ws.leaf_segs.push_back({node_k, beg, end});
                nr.first = -1;
                continue;  // degenerate: all obs in one bin
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

            // ---------------------------------------------------------------
            // Grow tree node + O(n_k) obs-list partition
            // ---------------------------------------------------------------
            tree.grow(node_k, chosen_feat, thresh);

            const uint8_t* col = Xq.data.data() + chosen_feat * Xq.n;
            ws.right_buf.clear();
            ws.right_buf.reserve(n_k);
            int write = beg;

#ifdef __ARM_NEON
            // Branchless NEON compress partition (gated on n_k >= 16).
            // Processes 8 obs per iteration: gathers bin values, compares
            // against thresh as a vector, extracts a scalar bitmask, then
            // routes left/right obs via the compress8 lookup table.
            // Eliminates data-dependent branch mispredictions (~50% miss
            // rate for balanced splits).
            if (n_k >= 16) {
                static const uint8_t pow2[8] = {1, 2, 4, 8, 16, 32, 64, 128};
                uint8x8_t vpow2   = vld1_u8(pow2);
                uint8x8_t vthresh = vdup_n_u8(thresh);
                constexpr int PPF = 12;

                int k = beg;
                for (; k + 7 < end; k += 8) {
                    if (k + PPF + 7 < end) {
                        __builtin_prefetch(col + ws.flat_obs[k + PPF],     0, 1);
                        __builtin_prefetch(col + ws.flat_obs[k + PPF + 4], 0, 1);
                    }
                    int obs[8];
                    uint8_t bins[8];
                    for (int i = 0; i < 8; i++) {
                        obs[i]  = ws.flat_obs[k + i];
                        bins[i] = col[obs[i]];
                    }
                    uint8x8_t vcmp = vcle_u8(vld1_u8(bins), vthresh);
                    uint8_t mask   = vaddv_u8(vand_u8(vcmp, vpow2));
                    int nl = compress8.cnt[mask];
                    const uint8_t* shuf = compress8.idx[mask];
                    for (int i = 0;  i < nl; i++) ws.flat_obs[write++]     = obs[shuf[i]];
                    for (int i = nl; i < 8;  i++) ws.right_buf.push_back(obs[shuf[i]]);
                }
                for (; k < end; k++) {
                    int ob = ws.flat_obs[k];
                    if (col[ob] <= thresh) ws.flat_obs[write++] = ob;
                    else                   ws.right_buf.push_back(ob);
                }
            } else {
                for (int k = beg; k < end; k++) {
                    int ob = ws.flat_obs[k];
                    if (col[ob] <= thresh) ws.flat_obs[write++] = ob;
                    else                   ws.right_buf.push_back(ob);
                }
            }
#else
            for (int k = beg; k < end; k++) {
                int ob = ws.flat_obs[k];
                if (col[ob] <= thresh) ws.flat_obs[write++] = ob;
                else                   ws.right_buf.push_back(ob);
            }
#endif // __ARM_NEON

            int left_end = write;
            for (int ob : ws.right_buf) ws.flat_obs[write++] = ob;

            nr.first = -1;
            if (left_end > beg)   ws.node_range[2 * node_k]     = {beg, left_end};
            if (write > left_end) ws.node_range[2 * node_k + 1] = {left_end, end};

            next_level.push_back(2 * node_k);
            next_level.push_back(2 * node_k + 1);
        }
        current_level = std::move(next_level);
    }
}

// -----------------------------------------------------------------------
// gfr_sweep
// -----------------------------------------------------------------------

void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees, n = state.n;

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) state.residual[i] += state.pred[t][i];

        grow_tree_gfr(state.trees[t], state.Xq, state.residual.data(),
                      n, state.p, state.sigma2, cfg, rng,
                      state.gfr_hist_ws, state.thread_pool.get());

        // Rebuild leaf state from GFR workspace — no re-traversal needed.
        // grow_tree_gfr populated ws.leaf_segs with (node, beg, end) for
        // each leaf; ws.flat_obs is already partitioned in matching order.
        auto& ws = state.gfr_hist_ws;
        auto& lc = state.leaf_counts[t];
        auto& fo = state.flat_obs[t];
        auto& ls = state.leaf_start[t];
        auto& li = state.leaf_indices[t];
        int full_size = state.trees[t].full_size;

        // Step 1: copy partitioned obs list from workspace
        std::copy(ws.flat_obs.begin(), ws.flat_obs.begin() + n, fo.begin());

        // Step 2: derive lc and li from leaf_segs (O(n) scattered write,
        // no tree traversal)
        std::fill(lc.begin(), lc.end(), 0);
        for (auto& seg : ws.leaf_segs) {
            lc[seg.node] = seg.end - seg.beg;
            for (int idx = seg.beg; idx < seg.end; idx++)
                li[fo[idx]] = seg.node;
        }

        // Step 3: exclusive prefix sum of lc → ls
        int running = 0;
        for (int k = 1; k <= full_size; k++) { ls[k] = running; running += lc[k]; }

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
