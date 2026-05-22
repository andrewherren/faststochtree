#include "MetalContext.h"
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
// Output layout: sum[feat * 256 + bin], cnt[feat * 256 + bin]
// ---------------------------------------------------------------------------
static void cpu_histogram(const uint8_t* Xq, const float* resid,
                          const int* obs_list, int n, int p, int n_k,
                          float* sum, int* cnt)
{
    memset(sum, 0, p * 256 * sizeof(float));
    memset(cnt, 0, p * 256 * sizeof(int));
    for (int j = 0; j < p; j++) {
        const uint8_t* col = Xq + j * n;
        float* sh = sum + j * 256;
        int*   ch = cnt + j * 256;
        for (int k = 0; k < n_k; k++) {
            int obs = obs_list[k];
            sh[col[obs]] += resid[obs];
            ch[col[obs]]++;
        }
    }
}

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------
using us_clock = std::chrono::high_resolution_clock;
using us_t     = std::chrono::duration<double, std::micro>;

static double elapsed_us(us_clock::time_point t0, us_clock::time_point t1) {
    return us_t(t1 - t0).count();
}

// Run fn() N times, return minimum elapsed microseconds.
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
// Noop section
// ---------------------------------------------------------------------------
static void run_noop_bench(gpu::MetalContext& ctx) {
    printf("--- warmup (includes shader compile) ---\n");
    ctx.noop_roundtrip_us();
    printf("\n--- noop kernel roundtrip (5 runs) ---\n");
    for (int i = 0; i < 5; ++i)
        ctx.noop_roundtrip_us();
}

// ---------------------------------------------------------------------------
// Histogram benchmark — one scenario
// ---------------------------------------------------------------------------
static void run_histogram_scenario(gpu::MetalContext& ctx,
                                   int n_k, int p,
                                   std::mt19937& rng)
{
    const int n = n_k;  // single root node owns all observations

    // Generate synthetic data.
    std::vector<uint8_t> Xq(n * p);
    std::vector<float>   resid(n);
    std::vector<int>     obs_list(n_k);

    std::uniform_int_distribution<int> udist(0, 255);
    std::normal_distribution<float>    ndist(0.f, 1.f);

    for (auto& v : Xq)    v = (uint8_t)udist(rng);
    for (auto& v : resid) v = ndist(rng);
    std::iota(obs_list.begin(), obs_list.end(), 0);

    const int hist_elems = p * 256;
    std::vector<float> cpu_sum(hist_elems), gpu_sum(hist_elems);
    std::vector<int>   cpu_cnt(hist_elems), gpu_cnt(hist_elems);

    // CPU: time over 5 runs, take min.
    double cpu_us = min_time_us(5, [&]{
        cpu_histogram(Xq.data(), resid.data(), obs_list.data(),
                      n, p, n_k, cpu_sum.data(), cpu_cnt.data());
    });

    // GPU: 1 warmup + 5 measured, take min kernel and corresponding host time.
    // Warmup here is per-scenario (not the initial shader compile warmup).
    ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(),
                        n, p, n_k, gpu_sum.data(), gpu_cnt.data());

    double best_kernel = 1e18, best_host = 1e18;
    for (int i = 0; i < 5; i++) {
        auto r = ctx.histogram_build(Xq.data(), resid.data(), obs_list.data(),
                                     n, p, n_k, gpu_sum.data(), gpu_cnt.data());
        if (r.gpu_kernel_us < best_kernel) {
            best_kernel = r.gpu_kernel_us;
            best_host   = r.host_us;
        }
    }

    // Correctness: max abs diff in sums (float order differs), exact match for counts.
    float max_diff = 0.f;
    int   cnt_fail = 0;
    for (int i = 0; i < hist_elems; i++) {
        max_diff = std::max(max_diff, std::abs(gpu_sum[i] - cpu_sum[i]));
        if (gpu_cnt[i] != cpu_cnt[i]) cnt_fail++;
    }

    // Speedup relative to GPU host roundtrip (the practical number).
    double speedup_host   = cpu_us / best_host;
    double speedup_kernel = cpu_us / best_kernel;

    printf("n_k=%7d  p=%3d:  cpu=%6.0f µs  gpu_kernel=%5.0f µs"
           "  host=%6.0f µs  kernel_speedup=%4.1fx  host_speedup=%4.2fx"
           "  diff=%.2e  cnt=%s\n",
           n_k, p,
           cpu_us, best_kernel, best_host,
           speedup_kernel, speedup_host,
           max_diff,
           cnt_fail == 0 ? "ok" : "FAIL");
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

    printf("\n=== histogram build (single node, CPU vs GPU) ===\n");
    printf("Each scenario: 1 warmup + 5 measured runs; CPU = min of 5.\n");
    printf("kernel_speedup = cpu / gpu_kernel   (ignores dispatch overhead)\n");
    printf("host_speedup   = cpu / gpu_host     (practical wall-clock)\n\n");

    std::mt19937 rng(42);

    for (int n_k : {1000, 10000, 100000}) {
        for (int p : {10, 50, 100}) {
            run_histogram_scenario(ctx, n_k, p, rng);
        }
        printf("\n");
    }

    return 0;
}
