#include "GFRAccel.h"
#include "faststochtree/model.hpp"
#include "faststochtree/mcmc.hpp"
#include "faststochtree/quantize.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <numeric>
#include <stdexcept>
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
    const int m = (cfg.p_eval > 0 && cfg.p_eval < p) ? cfg.p_eval : p;

    std::vector<float>    gpu_sum;
    std::vector<int>      gpu_cnt;
    std::vector<float>    gpu_feat_log;
    std::vector<int>      node_feat_orders;  // [n_active * m] per BFS level
    std::vector<int>      feat_idx(p);       // Fisher-Yates workspace
    std::vector<float>    log_split_ratio_buf;
    std::vector<unsigned> rng_seeds_buf;
    std::vector<MetalContext::SplitResult> split_results;

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

        // Fisher-Yates: generate per-node feature orderings before GPU dispatch.
        const int* feat_order_ptr = nullptr;
        if (m < p) {
            node_feat_orders.resize(n_active * m);
            for (int ai = 0; ai < n_active; ai++) {
                std::iota(feat_idx.begin(), feat_idx.end(), 0);
                for (int k = 0; k < m; k++) {
                    int j = k + rng.randint(0, p - k);
                    std::swap(feat_idx[k], feat_idx[j]);
                }
                std::copy(feat_idx.begin(), feat_idx.begin() + m,
                          node_feat_orders.data() + ai * m);
            }
            feat_order_ptr = node_feat_orders.data();
        }

        // Per-node log_split_ratio and xorshift32 seeds — computed before dispatch
        // so the GPU can sample without any CPU round-trip in between.
        log_split_ratio_buf.resize(n_active);
        rng_seeds_buf.resize(n_active);
        for (int ai = 0; ai < n_active; ai++) {
            int depth = bart::Tree::depth_of(active[ai].node_k);
            float p_split = cfg.alpha / std::pow(1.f + depth, cfg.beta);
            p_split = std::min(p_split, 1.f - 1e-6f);
            log_split_ratio_buf[ai] = std::log(p_split) - std::log(1.f - p_split);
            rng_seeds_buf[ai] = static_cast<unsigned>(rng.randint(0, INT_MAX)) + 1u;
        }

        gpu_sum.resize((size_t)n_active * m * 256);
        gpu_cnt.resize((size_t)n_active * m * 256);
        gpu_feat_log.resize((size_t)n_active * m);

        gpu_ctx.histogram_and_scan(
            Xq.data.data(), resid,
            ws.flat_obs.data(), n,
            node_ranges_buf.data(), n, p, n_active,
            m, feat_order_ptr,
            sigma2, cfg.leaf_prior_var, cfg.min_samples_leaf,
            gpu_sum.data(), gpu_cnt.data(), gpu_feat_log.data());

        split_results.resize(n_active);
        gpu_ctx.select_splits(
            gpu_feat_log.data(),
            gpu_sum.data(), gpu_cnt.data(),
            log_split_ratio_buf.data(), rng_seeds_buf.data(),
            n_active, m,
            sigma2, cfg.leaf_prior_var, cfg.min_samples_leaf,
            feat_order_ptr,
            split_results.data());

        // Per-node: use GPU sampling decision; CPU obs partition (unchanged).
        for (int ai = 0; ai < n_active; ai++) {
            int node_k = active[ai].node_k;
            int beg    = active[ai].beg;
            int end    = active[ai].end;

            const MetalContext::SplitResult& sr = split_results[ai];
            if (sr.feat == -1) {
                ws.leaf_segs.push_back({node_k, beg, end});
                ws.node_range[node_k].first = -1;
                continue;
            }

            tree.grow(node_k, sr.feat, sr.thresh);

            const uint8_t* col = Xq.data.data() + sr.feat * Xq.n;
            ws.right_buf.clear();
            int write = beg;
            for (int k = beg; k < end; k++) {
                int ob = ws.flat_obs[k];
                if (col[ob] <= sr.thresh) ws.flat_obs[write++] = ob;
                else                      ws.right_buf.push_back(ob);
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

// ── High-level fit functions ─────────────────────────────────────────────────

namespace {

static bart::BARTModel assemble_bart_model(
    std::vector<std::vector<float>>& test_vecs,
    std::vector<float>&              sigma2_vec,
    std::vector<bart::Forest>&       forests,
    bart::QuantizedX                 cuts,
    int n_test)
{
    bart::BARTModel m;
    m.n_samples      = static_cast<int>(forests.size());
    m.n_test         = n_test;
    m.forests        = std::move(forests);
    m.train_cuts     = std::move(cuts);
    m.sigma2_samples = std::move(sigma2_vec);
    m.test_samples.resize(static_cast<size_t>(m.n_samples) * m.n_test);
    for (int s = 0; s < m.n_samples; s++)
        if (m.n_test > 0)
            std::copy(test_vecs[s].begin(), test_vecs[s].end(),
                      m.test_samples.data() + s * m.n_test);
    return m;
}

} // anonymous namespace

bart::BARTModel fit_xbart_gpu(const float* X, const float* y, int n, int p,
                              const float* X_test, int n_test,
                              const bart::BARTConfig& cfg,
                              int n_burnin, int n_samples, int seed)
{
    MetalContext gpu_ctx;
    if (!gpu_ctx.ok())
        throw std::runtime_error("Metal GPU not available on this system.");

    bart::RNG rng(static_cast<unsigned>(seed));

    bart::BARTState state;
    state.n  = n;
    state.p  = p;
    state.Xq = bart::quantize(X, n, p);
    state.y  = y;
    bart::init_state(state, cfg, rng);
    state.gfr_hist_ws.alloc(n, p, state.trees[0].full_size);

    bart::QuantizedX test_qx;
    if (n_test > 0 && X_test)
        test_qx = bart::quantize_with_cuts(X_test, n_test, state.Xq);

    for (int s = 0; s < n_burnin; s++)
        gfr_sweep_gpu(state, cfg, rng, gpu_ctx);

    std::vector<std::vector<float>> test_vecs;
    std::vector<float>              sigma2_vec;
    std::vector<bart::Forest>       forests;
    test_vecs.reserve(n_samples);
    sigma2_vec.reserve(n_samples);
    forests.reserve(n_samples);

    for (int s = 0; s < n_samples; s++) {
        gfr_sweep_gpu(state, cfg, rng, gpu_ctx);

        if (n_test > 0) {
            std::vector<float> test_pred(n_test, 0.f);
            for (int t = 0; t < cfg.num_trees; t++)
                for (int i = 0; i < n_test; i++)
                    test_pred[i] += state.trees[t].leaf_value[
                        state.trees[t].traverse(test_qx.data.data(), i, n_test)];
            test_vecs.push_back(std::move(test_pred));
        }

        sigma2_vec.push_back(state.sigma2);
        forests.push_back(state.trees);
    }

    bart::QuantizedX cuts = state.Xq;
    cuts.data.clear();
    return assemble_bart_model(test_vecs, sigma2_vec, forests, std::move(cuts), n_test);
}

bart::BARTModel fit_warmstart_bart_gpu(const float* X, const float* y, int n, int p,
                                       const float* X_test, int n_test,
                                       const bart::BARTConfig& cfg,
                                       int n_gfr_burnin, int n_mcmc_burnin,
                                       int n_samples, int seed,
                                       bool keep_gfr_samples)
{
    MetalContext gpu_ctx;
    if (!gpu_ctx.ok())
        throw std::runtime_error("Metal GPU not available on this system.");

    bart::RNG rng(static_cast<unsigned>(seed));

    bart::BARTState state;
    state.n  = n;
    state.p  = p;
    state.Xq = bart::quantize(X, n, p);
    state.y  = y;
    bart::init_state(state, cfg, rng);
    state.gfr_hist_ws.alloc(n, p, state.trees[0].full_size);

    bart::QuantizedX test_qx;
    if (n_test > 0 && X_test)
        test_qx = bart::quantize_with_cuts(X_test, n_test, state.Xq);

    std::vector<std::vector<float>> test_vecs;
    std::vector<float>              sigma2_vec;
    std::vector<bart::Forest>       forests;
    int total = n_samples + (keep_gfr_samples ? n_gfr_burnin : 0);
    test_vecs.reserve(total);
    sigma2_vec.reserve(total);
    forests.reserve(total);

    auto collect = [&]() {
        if (n_test > 0) {
            std::vector<float> test_pred(n_test, 0.f);
            for (int t = 0; t < cfg.num_trees; t++)
                for (int i = 0; i < n_test; i++)
                    test_pred[i] += state.trees[t].leaf_value[
                        state.trees[t].traverse(test_qx.data.data(), i, n_test)];
            test_vecs.push_back(std::move(test_pred));
        }
        sigma2_vec.push_back(state.sigma2);
        forests.push_back(state.trees);
    };

    for (int s = 0; s < n_gfr_burnin; s++) {
        gfr_sweep_gpu(state, cfg, rng, gpu_ctx);
        if (keep_gfr_samples) collect();
    }

    for (int s = 0; s < n_mcmc_burnin; s++)
        bart::mcmc_sweep(state, cfg, rng);

    for (int s = 0; s < n_samples; s++) {
        bart::mcmc_sweep(state, cfg, rng);
        collect();
    }

    bart::QuantizedX cuts = state.Xq;
    cuts.data.clear();
    return assemble_bart_model(test_vecs, sigma2_vec, forests, std::move(cuts), n_test);
}

} // namespace gpu
