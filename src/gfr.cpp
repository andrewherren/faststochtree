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

void grow_tree_gfr(Tree& tree, const QuantizedX& Xq, const float* resid,
                   int n, int p, float sigma2, const BARTConfig& cfg, RNG& rng) {
    tree.reset();

    // Per-node observation index lists. Root (node 1) starts with all obs.
    std::unordered_map<int, std::vector<int>> node_obs;
    node_obs[1].resize(n);
    std::iota(node_obs[1].begin(), node_obs[1].end(), 0);

    // DFS-order queue: children pushed front so we can erase node_obs eagerly.
    std::deque<int> queue;
    queue.push_back(1);

    while (!queue.empty()) {
        int node_k = queue.front(); queue.pop_front();
        auto it = node_obs.find(node_k);
        if (it == node_obs.end()) continue;
        std::vector<int> obs = std::move(it->second);
        node_obs.erase(it);
        int n_k = (int)obs.size();

        int depth = Tree::depth_of(node_k);
        if (n_k < 2 * cfg.min_samples_leaf || depth >= tree.depth - 1) continue;

        // Total sufficient stats for this node
        float sum_T = 0.f;
        for (int i : obs) sum_T += resid[i];
        int count_T = n_k;

        // Build flat candidate list — naive: sort per feature per node
        struct Cand { int feat; uint8_t thresh; float sum_L; int count_L; };
        std::vector<Cand>  cands;
        std::vector<float> log_wts;

        std::vector<std::pair<uint8_t, float>> xr;
        xr.reserve(n_k);
        for (int j = 0; j < p; j++) {
            xr.clear();
            for (int i : obs) xr.push_back({Xq.at(i, j), resid[i]});
            std::sort(xr.begin(), xr.end());

            float sum_L = 0.f; int count_L = 0;
            for (int k = 0; k < n_k - 1; k++) {
                sum_L += xr[k].second; count_L++;
                if (xr[k].first >= xr[k+1].first) continue;  // skip ties
                int count_R = count_T - count_L;
                if (count_L < cfg.min_samples_leaf || count_R < cfg.min_samples_leaf) continue;
                float log_ml = leaf_log_ml(sum_L, count_L, sigma2, cfg.leaf_prior_var)
                             + leaf_log_ml(sum_T - sum_L, count_R, sigma2, cfg.leaf_prior_var);
                cands.push_back({j, xr[k].first, sum_L, count_L});
                log_wts.push_back(log_ml);
            }
        }

        // No-split option weighted by prior and number of split candidates
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
        if (c.feat == -1) continue;  // no-split chosen

        tree.grow(node_k, c.feat, c.thresh);

        std::vector<int> lo, ro;
        lo.reserve(c.count_L); ro.reserve(n_k - c.count_L);
        for (int i : obs)
            (Xq.at(i, c.feat) <= c.thresh ? lo : ro).push_back(i);

        node_obs[2 * node_k]     = std::move(lo);
        node_obs[2 * node_k + 1] = std::move(ro);
        queue.push_front(2 * node_k + 1);
        queue.push_front(2 * node_k);
    }
}

void gfr_sweep(BARTState& state, const BARTConfig& cfg, RNG& rng) {
    int T = cfg.num_trees, n = state.n;
    const uint8_t* xq = state.Xq.data.data();

    for (int t = 0; t < T; t++) {
        for (int i = 0; i < n; i++) state.residual[i] += state.pred[t][i];

        grow_tree_gfr(state.trees[t], state.Xq, state.residual.data(),
                      n, state.p, state.sigma2, cfg, rng);

        // Rebuild leaf index cache — tree was rebuilt from scratch
        for (int i = 0; i < n; i++)
            state.leaf_indices[t][i] = state.trees[t].traverse(xq, i, n);

        // zeros as pred_off: residual is already the full partial residual
        sample_leaves(state.trees[t], state.residual.data(), state.ws.zeros.data(),
                      n, state.sigma2, cfg, rng, state.leaf_indices[t], state.ws);

        for (int i = 0; i < n; i++)
            state.pred[t][i] = state.trees[t].leaf_value[state.leaf_indices[t][i]];
        for (int i = 0; i < n; i++) state.residual[i] -= state.pred[t][i];
    }
    sample_sigma2(state.residual.data(), n, state.sigma2, cfg, rng);
}

} // namespace bart
