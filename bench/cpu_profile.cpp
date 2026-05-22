// cpu_profile.cpp
// Times each phase of a GFR node split independently.
//
// Phases (matching grow_tree_gfr):
//   sum_T     — residual accumulation over the node            O(n_k)
//   hist      — histogram accumulation only (all m features)   O(n_k * m)
//   scan      — prefix scan over bins (all m features)         O(256 * m)
//   stage1    — feature softmax + categorical sample           O(m)
//   stage2    — cutpoint re-scan + categorical sample          O(256)
//   partition — obs-list split into left / right               O(n_k)
//
// Note: in gfr.cpp, hist and scan are fused per-feature for cache locality.
// Separating them here means scan sees slightly cooler cache than in prod,
// but since scan is O(256) vs O(n_k) for hist, the separation effect is small.

#include "faststochtree/model.hpp"
#include "faststochtree/quantize.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#ifdef __ARM_NEON
#include <arm_neon.h>
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
#endif

using us_clock = std::chrono::high_resolution_clock;
using us_dur   = std::chrono::duration<double, std::micro>;

static double elapsed_us(us_clock::time_point t0, us_clock::time_point t1) {
    return us_dur(t1 - t0).count();
}

template<typename Fn>
static double min_time_us(int N, Fn fn) {
    double best = 1e18;
    for (int i = 0; i < N; i++) {
        auto t0 = us_clock::now();
        fn();
        auto t1 = us_clock::now();
        best = std::min(best, elapsed_us(t0, t1));
    }
    return best;
}

// ---------------------------------------------------------------------------

struct PhaseResult {
    double sum_T_us;
    double hist_us;
    double scan_us;
    double stage1_us;
    double stage2_us;
    double partition_us;
    double total_us() const {
        return sum_T_us + hist_us + scan_us + stage1_us + stage2_us + partition_us;
    }
};

static PhaseResult profile_node(int n_k, int p, std::mt19937& rng) {
    // ---- synthetic data ------------------------------------------------
    // Generate float X[n*p], quantize to uint8 column-major.
    const int n = n_k;
    std::uniform_real_distribution<float> udist(0.f, 1.f);
    std::normal_distribution<float>       ndist(0.f, 1.f);

    std::vector<float> X_raw(n * p);
    for (auto& v : X_raw) v = udist(rng);
    bart::QuantizedX Xq = bart::quantize(X_raw.data(), n, p, 255);

    std::vector<float> resid(n);
    for (auto& v : resid) v = ndist(rng);

    // Single root node — all observations in [0, n_k).
    std::vector<int> obs_list(n_k);
    std::iota(obs_list.begin(), obs_list.end(), 0);

    // Workspace buffers (same layout as GFRHistWorkspace).
    std::vector<float> sum_hists(p * 256);
    std::vector<int>   cnt_hists(p * 256);
    std::vector<float> feat_log_total(p + 1);

    // Identity feature order (no Fisher-Yates, for reproducibility).
    std::vector<int> feat_order(p);
    std::iota(feat_order.begin(), feat_order.end(), 0);

    bart::BARTConfig cfg;
    cfg.leaf_prior_var = 1.f / cfg.num_trees;  // calibrated default
    const int   m              = p;
    const int   beg            = 0;
    const int   end            = n_k;
    const int   count_T        = n_k;
    const float sigma2         = 1.f;
    const int   min_samp       = cfg.min_samples_leaf;
    const float NEG_INF        = -std::numeric_limits<float>::infinity();
    const int   REPS           = 5;

    // ---- phase 1: sum_T ------------------------------------------------
    volatile float sum_T_sink = 0.f;
    double sum_T_us = min_time_us(REPS, [&]{
        float sum_T = 0.f;
        for (int k = beg; k < end; k++) sum_T += resid[obs_list[k]];
        sum_T_sink = sum_T;
    });
    float sum_T = sum_T_sink;

    // ---- phase 2: hist (accumulation only) -----------------------------
    // Replicates the inner scatter loop of hist_build without the prefix scan.
    double hist_us = min_time_us(REPS, [&]{
        for (int fi = 0; fi < m; fi++) {
            int j = feat_order[fi];
            float* sh = sum_hists.data() + fi * 256;
            int*   ch = cnt_hists.data() + fi * 256;
            std::fill(sh, sh + 256, 0.f);
            std::fill(ch, ch + 256, 0);
            const uint8_t* col = Xq.data.data() + j * n;
            constexpr int PF = 12;
            int hk = beg;
            if (n_k >= 128) {
                for (; hk + 3 < end; hk += 4) {
                    #ifdef __GNUC__
                    if (hk + PF + 3 < end) {
                        __builtin_prefetch(col + obs_list[hk + PF],     0, 1);
                        __builtin_prefetch(col + obs_list[hk + PF + 1], 0, 1);
                        __builtin_prefetch(col + obs_list[hk + PF + 2], 0, 1);
                        __builtin_prefetch(col + obs_list[hk + PF + 3], 0, 1);
                        __builtin_prefetch(resid.data() + obs_list[hk + PF],     0, 1);
                        __builtin_prefetch(resid.data() + obs_list[hk + PF + 1], 0, 1);
                        __builtin_prefetch(resid.data() + obs_list[hk + PF + 2], 0, 1);
                        __builtin_prefetch(resid.data() + obs_list[hk + PF + 3], 0, 1);
                    }
                    #endif
                    int o0 = obs_list[hk], o1 = obs_list[hk+1];
                    int o2 = obs_list[hk+2], o3 = obs_list[hk+3];
                    uint8_t b0 = col[o0], b1 = col[o1];
                    uint8_t b2 = col[o2], b3 = col[o3];
                    sh[b0] += resid[o0]; ch[b0]++;
                    sh[b1] += resid[o1]; ch[b1]++;
                    sh[b2] += resid[o2]; ch[b2]++;
                    sh[b3] += resid[o3]; ch[b3]++;
                }
            }
            for (; hk < end; hk++) {
                int obs = obs_list[hk];
                uint8_t bin = col[obs];
                sh[bin] += resid[obs];
                ch[bin]++;
            }
        }
    });
    // Rebuild histogram once more so scan sees valid data.
    for (int fi = 0; fi < m; fi++) {
        int j = feat_order[fi];
        float* sh = sum_hists.data() + fi * 256;
        int*   ch = cnt_hists.data() + fi * 256;
        std::fill(sh, sh + 256, 0.f);
        std::fill(ch, ch + 256, 0);
        const uint8_t* col = Xq.data.data() + j * n;
        for (int k = beg; k < end; k++) {
            int obs = obs_list[k]; uint8_t bin = col[obs];
            sh[bin] += resid[obs]; ch[bin]++;
        }
    }

    // ---- phase 3: scan (prefix scan only, hist already built) ----------
    volatile float scan_sink = 0.f;
    double scan_us = min_time_us(REPS, [&]{
        float acc = 0.f;
        for (int fi = 0; fi < m; fi++) {
            float* sh = sum_hists.data() + fi * 256;
            int*   ch = cnt_hists.data() + fi * 256;
            float sum_L = 0.f; int count_L = 0;
            float max_lw = NEG_INF, sum_exp = 0.f;
            for (int b = 0; b < 255; b++) {
                if (ch[b] == 0) continue;
                sum_L   += sh[b];
                count_L += ch[b];
                int count_R = count_T - count_L;
                if (count_L < min_samp || count_R < min_samp) continue;
                float log_ml =
                    bart::leaf_log_ml(sum_L, (float)count_L, sigma2, cfg.leaf_prior_var)
                  + bart::leaf_log_ml(sum_T - sum_L, (float)count_R, sigma2, cfg.leaf_prior_var);
                if (log_ml > max_lw) {
                    sum_exp = sum_exp * std::exp(max_lw - log_ml) + 1.f;
                    max_lw = log_ml;
                } else {
                    sum_exp += std::exp(log_ml - max_lw);
                }
            }
            feat_log_total[fi] = (sum_exp > 0.f) ? max_lw + std::log(sum_exp) : NEG_INF;
            acc += feat_log_total[fi];
        }
        scan_sink = acc;
    });

    // ---- phase 4: stage1 (feature softmax + sample) --------------------
    // Set no-split option (index m).
    feat_log_total[m] = NEG_INF;
    volatile int chosen_fi_sink = 0;
    double stage1_us = min_time_us(REPS, [&]{
        float max_ft = *std::max_element(feat_log_total.data(), feat_log_total.data() + m + 1);
        float total_ft = 0.f;
        for (int fi = 0; fi <= m; fi++) total_ft += std::exp(feat_log_total[fi] - max_ft);
        float u = 0.5f * total_ft, cum = 0.f;
        int chosen = m;
        for (int fi = 0; fi < m; fi++) {
            cum += std::exp(feat_log_total[fi] - max_ft);
            if (u <= cum) { chosen = fi; break; }
        }
        chosen_fi_sink = chosen;
    });
    int chosen_fi = (chosen_fi_sink < m) ? chosen_fi_sink : 0;

    // ---- phase 5: stage2 (cutpoint re-scan + sample) -------------------
    volatile uint8_t thresh_sink = 0;
    double stage2_us = min_time_us(REPS, [&]{
        float* sh = sum_hists.data() + chosen_fi * 256;
        int*   ch = cnt_hists.data() + chosen_fi * 256;
        float sum_L = 0.f; int count_L = 0;
        float max_lw2 = NEG_INF, total_ct = 0.f;
        uint8_t best_thresh = 0;
        for (int b = 0; b < 255; b++) {
            if (ch[b] == 0) continue;
            sum_L   += sh[b];
            count_L += ch[b];
            int count_R = count_T - count_L;
            if (count_L < min_samp || count_R < min_samp) continue;
            float log_ml =
                bart::leaf_log_ml(sum_L, (float)count_L, sigma2, cfg.leaf_prior_var)
              + bart::leaf_log_ml(sum_T - sum_L, (float)count_R, sigma2, cfg.leaf_prior_var);
            float w = std::exp(log_ml - max_lw2);
            if (log_ml > max_lw2) {
                total_ct = total_ct * std::exp(max_lw2 - log_ml) + 1.f;
                max_lw2 = log_ml;
            } else {
                total_ct += w;
            }
            if (w >= 0.5f * total_ct) best_thresh = (uint8_t)b;
        }
        thresh_sink = best_thresh;
    });
    uint8_t thresh = thresh_sink ? thresh_sink : 127;

    // ---- phase 6: partition (obs-list split) ---------------------------
    std::vector<int>     flat_obs(obs_list);
    std::vector<int>     right_buf;
    right_buf.reserve(n_k);
    int chosen_feat = feat_order[chosen_fi];
    const uint8_t* part_col = Xq.data.data() + chosen_feat * n;
    volatile int left_end_sink = 0;

    double partition_us = min_time_us(REPS, [&]{
        std::copy(obs_list.begin(), obs_list.end(), flat_obs.begin());
        right_buf.clear();
        int write = beg;
#ifdef __ARM_NEON
        if (n_k >= 16) {
            static const uint8_t pow2[8] = {1,2,4,8,16,32,64,128};
            uint8x8_t vpow2   = vld1_u8(pow2);
            uint8x8_t vthresh = vdup_n_u8(thresh);
            constexpr int PPF = 12;
            int k = beg;
            for (; k + 7 < end; k += 8) {
                if (k + PPF + 7 < end) {
                    __builtin_prefetch(part_col + flat_obs[k + PPF],     0, 1);
                    __builtin_prefetch(part_col + flat_obs[k + PPF + 4], 0, 1);
                }
                int obs[8]; uint8_t bins[8];
                for (int i = 0; i < 8; i++) { obs[i] = flat_obs[k+i]; bins[i] = part_col[obs[i]]; }
                uint8x8_t vcmp = vcle_u8(vld1_u8(bins), vthresh);
                uint8_t mask   = vaddv_u8(vand_u8(vcmp, vpow2));
                int nl = compress8.cnt[mask];
                const uint8_t* shuf = compress8.idx[mask];
                for (int i = 0;  i < nl; i++) flat_obs[write++]        = obs[shuf[i]];
                for (int i = nl; i < 8;  i++) right_buf.push_back(obs[shuf[i]]);
            }
            for (; k < end; k++) {
                int ob = flat_obs[k];
                if (part_col[ob] <= thresh) flat_obs[write++] = ob;
                else right_buf.push_back(ob);
            }
        } else {
#endif
            for (int k = beg; k < end; k++) {
                int ob = flat_obs[k];
                if (part_col[ob] <= thresh) flat_obs[write++] = ob;
                else right_buf.push_back(ob);
            }
#ifdef __ARM_NEON
        }
#endif
        left_end_sink = write;
    });

    return PhaseResult{sum_T_us, hist_us, scan_us, stage1_us, stage2_us, partition_us};
}

// ---------------------------------------------------------------------------

static void print_bar(double frac, int width = 20) {
    int filled = (int)(frac * width + 0.5);
    for (int i = 0; i < filled; i++) putchar('#');
    for (int i = filled; i < width; i++) putchar('.');
}

int main() {
    printf("=== GFR node split: CPU phase breakdown ===\n");
    printf("Single root node (n_k = n). Best of 5 runs per phase.\n");
    printf("Phases: sum_T | hist (accum) | scan (prefix) | stage1 (feat sample) |"
           " stage2 (cut sample) | partition\n\n");

    std::mt19937 rng(42);

    for (int n_k : {1000, 10000, 100000}) {
        for (int p : {10, 50, 100}) {
            auto r = profile_node(n_k, p, rng);
            double total = r.total_us();

            auto pct = [&](double v) { return 100.0 * v / total; };

            printf("n_k=%7d  p=%3d  total=%7.0f µs\n", n_k, p, total);
            printf("  sum_T    %6.0f µs (%4.1f%%)  ", r.sum_T_us,     pct(r.sum_T_us));
            print_bar(r.sum_T_us / total); printf("\n");
            printf("  hist     %6.0f µs (%4.1f%%)  ", r.hist_us,      pct(r.hist_us));
            print_bar(r.hist_us / total); printf("\n");
            printf("  scan     %6.0f µs (%4.1f%%)  ", r.scan_us,      pct(r.scan_us));
            print_bar(r.scan_us / total); printf("\n");
            printf("  stage1   %6.0f µs (%4.1f%%)  ", r.stage1_us,    pct(r.stage1_us));
            print_bar(r.stage1_us / total); printf("\n");
            printf("  stage2   %6.0f µs (%4.1f%%)  ", r.stage2_us,    pct(r.stage2_us));
            print_bar(r.stage2_us / total); printf("\n");
            printf("  partition%6.0f µs (%4.1f%%)  ", r.partition_us, pct(r.partition_us));
            print_bar(r.partition_us / total); printf("\n\n");
        }
    }

    return 0;
}
