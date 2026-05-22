#include "GFRAccel.h"
#include "MetalContext.h"
#include "faststochtree/mcmc.hpp"
#include "faststochtree/quantize.hpp"
#include "faststochtree/gfr.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

// ---------------------------------------------------------------------------
// CPU reference histogram build — matches the GFR inner loop exactly.
// Output layout: sum[(node*p + feat)*256 + bin], same as GPU.
// ---------------------------------------------------------------------------
static void cpu_histogram_node(const uint8_t* Xq, const float* resid,
                                const int* obs_list, int beg, int end,
                                int n, int p,
                                float* sum, int* cnt)
{
    memset(sum, 0, p * 256 * sizeof(float));
    memset(cnt, 0, p * 256 * sizeof(int));
    for (int j = 0; j < p; j++) {
        const uint8_t* col = Xq + j * n;
        float* sh = sum + j * 256;
        int*   ch = cnt + j * 256;
        for (int k = beg; k < end; k++) {
            int obs = obs_list[k];
            sh[col[obs]] += resid[obs];
            ch[col[obs]]++;
        }
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using clk = std::chrono::high_resolution_clock;
using us  = std::chrono::duration<double, std::micro>;

template<typename Fn>
static double min_time_us(int N, Fn fn) {
    double best = 1e18;
    for (int i = 0; i < N; i++) {
        auto t0 = clk::now();
        fn();
        best = std::min(best, us(clk::now() - t0).count());
    }
    return best;
}

// ---------------------------------------------------------------------------
// Section 1: noop roundtrip
// ---------------------------------------------------------------------------
static void run_noop_bench(gpu::MetalContext& ctx) {
    printf("--- warmup (includes shader compile) ---\n");
    ctx.noop_roundtrip_us();
    printf("\n--- noop kernel roundtrip (5 runs) ---\n");
    for (int i = 0; i < 5; ++i)
        ctx.noop_roundtrip_us();
}

// ---------------------------------------------------------------------------
// Section 2: single-node histogram sweep (Phase 1 baseline)
// ---------------------------------------------------------------------------
static void run_single_node_sweep(gpu::MetalContext& ctx, std::mt19937& rng) {
    printf("\n=== histogram build: single node, varying n_k and p ===\n");
    printf("1 warmup + 5 measured runs. CPU = min of 5.\n");
    printf("kernel_speedup = cpu / gpu_kernel   host_speedup = cpu / gpu_host\n\n");

    for (int n_k : {1000, 10000, 100000}) {
        for (int p : {10, 50, 100}) {
            const int n = n_k;
            std::vector<uint8_t> Xq(n * p);
            std::vector<float>   resid(n);
            std::vector<int>     obs_list(n_k);
            std::uniform_int_distribution<int> udist(0, 255);
            std::normal_distribution<float>    ndist(0.f, 1.f);
            for (auto& v : Xq)    v = (uint8_t)udist(rng);
            for (auto& v : resid) v = ndist(rng);
            std::iota(obs_list.begin(), obs_list.end(), 0);

            std::vector<float> cpu_sum(p * 256), gpu_sum(p * 256);
            std::vector<int>   cpu_cnt(p * 256), gpu_cnt(p * 256);

            double cpu_us = min_time_us(5, [&]{
                cpu_histogram_node(Xq.data(), resid.data(), obs_list.data(),
                                   0, n_k, n, p, cpu_sum.data(), cpu_cnt.data());
            });

            int ranges[2] = {0, n_k};
            // warmup
            ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(), n_k, ranges,
                                 n, p, 1, gpu_sum.data(), gpu_cnt.data());
            double best_k = 1e18, best_h = 1e18;
            for (int i = 0; i < 5; i++) {
                auto r = ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(),
                                             n_k, ranges, n, p, 1,
                                             gpu_sum.data(), gpu_cnt.data());
                if (r.gpu_kernel_us < best_k) { best_k = r.gpu_kernel_us; best_h = r.host_us; }
            }

            float max_diff = 0.f; int cnt_fail = 0;
            for (int i = 0; i < p * 256; i++) {
                max_diff = std::max(max_diff, std::abs(gpu_sum[i] - cpu_sum[i]));
                if (gpu_cnt[i] != cpu_cnt[i]) cnt_fail++;
            }

            printf("n_k=%7d  p=%3d:  cpu=%6.0f µs  gpu_kernel=%5.0f µs"
                   "  host=%6.0f µs  kernel=%4.1fx  host=%4.2fx"
                   "  diff=%.2e  cnt=%s\n",
                   n_k, p, cpu_us, best_k, best_h,
                   cpu_us / best_k, cpu_us / best_h,
                   max_diff, cnt_fail == 0 ? "ok" : "FAIL");
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Section 3: BFS-level sweep — find the GPU/CPU crossover depth
//
// Simulates a balanced tree at each BFS depth:
//   depth d → 2^d nodes, each with n/2^d observations.
// All nodes at a depth are dispatched in ONE GPU command buffer.
// CPU time is the sum over nodes (sequential, as in the real GFR loop).
// ---------------------------------------------------------------------------
static void run_bfs_sweep(gpu::MetalContext& ctx, std::mt19937& rng) {
    printf("\n=== BFS-level histogram: batched GPU dispatch vs sequential CPU ===\n");
    printf("n=100000, balanced tree (n_k halves each depth). 1 warmup + 5 runs.\n");
    printf("GPU dispatches ALL nodes at a depth in one command buffer.\n");
    printf("CPU runs nodes sequentially (as in grow_tree_gfr).\n\n");

    // Fixed n; 100000 divides cleanly through depth 5 (100000 / 32 = 3125).
    const int n = 100000;
    const int max_depth = 5;

    for (int p : {10, 50, 100}) {
        // Generate dataset once for this (n, p).
        std::vector<uint8_t> Xq(n * p);
        std::vector<float>   resid(n);
        std::vector<int>     obs_list(n);  // identity permutation
        std::uniform_int_distribution<int> udist(0, 255);
        std::normal_distribution<float>    ndist(0.f, 1.f);
        for (auto& v : Xq)    v = (uint8_t)udist(rng);
        for (auto& v : resid) v = ndist(rng);
        std::iota(obs_list.begin(), obs_list.end(), 0);

        printf("p=%3d\n", p);
        printf("  depth  nodes  n_k/node  cpu_level  gpu_kernel  gpu_host"
               "  kernel_spdup  host_spdup  verdict\n");

        for (int d = 0; d <= max_depth; d++) {
            int n_nodes = 1 << d;          // 2^d
            int n_k     = n / n_nodes;     // obs per node (balanced)

            // Build node_ranges: [{0,n_k}, {n_k,2*n_k}, ..., {(nodes-1)*n_k, n}]
            std::vector<int> node_ranges(n_nodes * 2);
            for (int i = 0; i < n_nodes; i++) {
                node_ranges[2*i]   = i * n_k;
                node_ranges[2*i+1] = (i == n_nodes - 1) ? n : (i+1) * n_k;
            }

            // Output buffers (contents not verified — we only need timing here).
            std::vector<float> gpu_sum(n_nodes * p * 256);
            std::vector<int>   gpu_cnt(n_nodes * p * 256);
            std::vector<float> cpu_sum(p * 256);
            std::vector<int>   cpu_cnt(p * 256);

            // CPU: run each node sequentially, accumulate total time.
            double cpu_level_us = min_time_us(5, [&]{
                for (int i = 0; i < n_nodes; i++) {
                    int beg = node_ranges[2*i], end = node_ranges[2*i+1];
                    cpu_histogram_node(Xq.data(), resid.data(), obs_list.data(),
                                       beg, end, n, p,
                                       cpu_sum.data(), cpu_cnt.data());
                }
            });

            // GPU: one batched dispatch for all nodes at this depth.
            // Warmup first.
            ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(), n,
                                 node_ranges.data(), n, p, n_nodes,
                                 nullptr, nullptr);
            double best_k = 1e18, best_h = 1e18;
            for (int i = 0; i < 5; i++) {
                auto r = ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(),
                                             n, node_ranges.data(), n, p, n_nodes,
                                             nullptr, nullptr);
                if (r.gpu_kernel_us < best_k) { best_k = r.gpu_kernel_us; best_h = r.host_us; }
            }

            double k_spdup = cpu_level_us / best_k;
            double h_spdup = cpu_level_us / best_h;
            const char* verdict = h_spdup >= 1.0 ? "GPU wins" : "CPU wins";

            printf("  d=%d    %4d   %7d   %7.0f µs  %7.0f µs  %7.0f µs"
                   "     %5.2fx       %5.2fx   %s\n",
                   d, n_nodes, n_k,
                   cpu_level_us, best_k, best_h,
                   k_spdup, h_spdup, verdict);
        }
        printf("\n");
    }
}

// ---------------------------------------------------------------------------
// Section 4: end-to-end GFR sweep — CPU gfr_sweep vs GPU gfr_sweep_gpu
//
// Uses two independent BARTState objects with identical data but separate RNGs
// so the two code paths are not serialized. Each state runs 1 warmup + 3
// measured sweeps; we report min time across measured runs.
//
// Constraints for GPU path:
//   - cfg.p_eval must be 0 (full feature set, no subsampling)
//   - constant leaf only (no BCF z vector)
// ---------------------------------------------------------------------------
static void run_gfr_sweep_bench(gpu::MetalContext& ctx) {
    printf("\n=== GFR sweep: bart::gfr_sweep vs gpu::gfr_sweep_gpu ===\n");

    const int n      = 100000;
    const int n_test = 0;

    for (int p : {10, 20}) {
        for (int T : {50, 200}) {
            // Generate random training data.
            std::mt19937 data_rng(123);
            std::uniform_real_distribution<float> xdist(-1.f, 1.f);
            std::normal_distribution<float>       ydist(0.f, 1.f);
            std::vector<float> X(n * p), y(n);
            for (auto& v : X) v = xdist(data_rng);
            for (auto& v : y) v = ydist(data_rng);

            bart::BARTConfig cfg;
            cfg.num_trees      = T;
            cfg.tree_depth     = 6;
            cfg.leaf_prior_var = (0.5f * 0.5f) / T;
            cfg.sigma2_shape   = 3.0f;
            cfg.sigma2_scale   = 1.0f;
            cfg.p_eval         = 0;  // GPU path requires full feature set

            // CPU state
            bart::RNG rng_cpu(42);
            bart::BARTState cpu_state;
            cpu_state.n  = n; cpu_state.p = p;
            cpu_state.Xq = bart::quantize(X.data(), n, p);
            cpu_state.y  = y.data();
            bart::init_state(cpu_state, cfg, rng_cpu);
            cpu_state.gfr_hist_ws.alloc(n, p, cpu_state.trees[0].full_size);

            // GPU state — same data, different RNG seed for independence
            bart::RNG rng_gpu(42);
            bart::BARTState gpu_state;
            gpu_state.n  = n; gpu_state.p = p;
            gpu_state.Xq = bart::quantize(X.data(), n, p);
            gpu_state.y  = y.data();
            bart::init_state(gpu_state, cfg, rng_gpu);
            gpu_state.gfr_hist_ws.alloc(n, p, gpu_state.trees[0].full_size);

            // Warmup
            bart::gfr_sweep(cpu_state, cfg, rng_cpu);
            gpu::gfr_sweep_gpu(gpu_state, cfg, rng_gpu, ctx);

            // Measured runs
            const int N_RUNS = 3;
            double best_cpu = 1e18, best_gpu = 1e18;
            for (int i = 0; i < N_RUNS; i++) {
                auto t0 = clk::now();
                bart::gfr_sweep(cpu_state, cfg, rng_cpu);
                best_cpu = std::min(best_cpu, us(clk::now() - t0).count());

                t0 = clk::now();
                gpu::gfr_sweep_gpu(gpu_state, cfg, rng_gpu, ctx);
                best_gpu = std::min(best_gpu, us(clk::now() - t0).count());
            }

            printf("n=%d  p=%2d  T=%3d:  cpu=%8.0f µs  gpu=%8.0f µs  speedup=%5.2fx\n",
                   n, p, T, best_cpu, best_gpu, best_cpu / best_gpu);
        }
    }
    printf("\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    printf("=== faststochtree GPU bench ===\n\n");

    gpu::MetalContext ctx;
    if (!ctx.ok()) {
        fprintf(stderr, "Metal not available — cannot run GPU bench.\n");
        return 1;
    }
    printf("Device: %s\n\n", ctx.device_name());

    run_noop_bench(ctx);

    std::mt19937 rng(42);
    run_single_node_sweep(ctx, rng);
    run_bfs_sweep(ctx, rng);
    run_gfr_sweep_bench(ctx);

    return 0;
}
