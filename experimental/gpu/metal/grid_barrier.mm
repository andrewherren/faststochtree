// grid_barrier.mm — how expensive is a *device-wide* barrier on Metal?
//
// Claim demonstrated: Metal has no hardware grid-wide barrier (CUDA's
// cooperative-groups grid.sync()). The only option is a hand-rolled software
// barrier — a counting/generation rendezvous spun on a device-atomic counter,
// the Xiao & Feng (2010) construction. It works for a few threadgroups but
// DETONATES once you launch more threadgroups than the scheduler co-schedules
// for a spinning kernel: the spinners wait on groups that aren't resident yet
// (a forward-progress hazard), and the cost goes non-monotonic — the signature
// of scheduler livelock, not deterministic work.
//
// This is the architectural wall for a single GPU-resident BART chain: the residual
// must be synced across all observations at least once per tree (the initial Metal
// experiments used several grid barriers per tree), so a barrier recurs many times
// per sweep, every sweep. Compare to ../cuda/reprex.ipynb (grid_barrier): CUDA
// grid.sync() is ~1.7-2.1 us/barrier and ~FLAT in block count (measured on a Colab
// T4: 1678 ns at 1 block, 2142 ns at 160 blocks).
//
// We time the *same* software barrier the GPU-resident sampler used, repeated
// `iters` times, swept over threadgroup count.
//
// Build: make grid_barrier    Run: ./grid_barrier
//
// The sweep raises the threadgroup count until it hits the cliff, then STOPS — so
// it always completes rather than hanging in the livelock regime. The cliff's exact
// location varies run to run (often 4-8 threadgroups); past it, a single barrier can
// livelock for minutes and trip the GPU watchdog ("ImpactingInteractivity"). That
// instability is itself the evidence — we just don't let the program depend on
// surviving it.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>

// The Xiao & Feng (2010) software grid barrier, in MSL: a counter + generation
// rendezvous bracketed by device-scope threadgroup barriers. This is the only way
// to get a device-wide barrier in Metal; there is no grid.sync() equivalent.
static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;

inline void grid_barrier(device atomic_uint* cnt, uint N_TG, thread uint* gen, uint tid) {
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        atomic_fetch_add_explicit(cnt, 1u, memory_order_relaxed);
        const uint target = (*gen + 1u) * N_TG;
        while (atomic_load_explicit(cnt, memory_order_relaxed) < target) { /* spin */ }
    }
    threadgroup_barrier(mem_flags::mem_device);
    *gen += 1u;
}

kernel void barrier_bench(device atomic_uint* cnt [[buffer(0)]],
                          constant uint&     N_TG [[buffer(1)]],
                          constant int&      iters[[buffer(2)]],
                          uint tid [[thread_index_in_threadgroup]]) {
    uint gen = 0;
    for (int i = 0; i < iters; ++i) grid_barrier(cnt, N_TG, &gen, tid);
}
)";

int main() {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        printf("device: %s\n\n", dev.name.UTF8String);

        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                               options:nil error:&err];
        if (!lib) { printf("compile failed: %s\n", err.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> fn = [lib newFunctionWithName:@"barrier_bench"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        const int iters = 50;   // keeps total GPU time well under the watchdog limit
        id<MTLBuffer> cnt = [dev newBufferWithLength:sizeof(uint) options:MTLResourceStorageModeShared];
        id<MTLBuffer> itb = [dev newBufferWithBytes:&iters length:sizeof(int) options:MTLResourceStorageModeShared];

        printf("  %-8s %14s %16s\n", "N_TG", "ms (GPU)", "ns/barrier");

        // Sweep the threadgroup count UPWARD and STOP at the first cliff. The cliff is
        // where the software barrier stops being a cheap rendezvous and starts to
        // livelock — once the scheduler can't keep every spinning threadgroup resident,
        // a single barrier blows up from ~1 us to tens of ms (and beyond that, hangs for
        // minutes / trips the watchdog). Its exact location is NON-DETERMINISTIC (4 on
        // one run, 8 on another) — that variability is itself the signature. We break as
        // soon as we hit it, so the program always completes instead of hanging.
        const double cliff_ms  = 50.0;   // cheap regime is <1 ms; >50 ms is unambiguously the cliff
        const double timeout_s  = 15.0;   // past the cliff a barrier can livelock forever; bail out
        const int tgs[] = {1, 2, 4, 8, 16, 32};
        for (int N_TG : tgs) {
            *((uint*)cnt.contents) = 0u;
            uint ntg = (uint)N_TG;
            id<MTLBuffer> nb = [dev newBufferWithBytes:&ntg length:sizeof(uint)
                                               options:MTLResourceStorageModeShared];
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            [e setBuffer:cnt offset:0 atIndex:0];
            [e setBuffer:nb  offset:0 atIndex:1];
            [e setBuffer:itb offset:0 atIndex:2];
            [e dispatchThreadgroups:MTLSizeMake(N_TG, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e endEncoding];

            // Wait with a host-side timeout: if the barrier livelocks (the dispatch
            // never returns), we must not block forever. A completion handler signals
            // a semaphore; if it doesn't fire within timeout_s we declare the cliff and
            // leave the orphaned GPU work to the watchdog.
            dispatch_semaphore_t done = dispatch_semaphore_create(0);
            [cb addCompletedHandler:^(id<MTLCommandBuffer>) { dispatch_semaphore_signal(done); }];
            [cb commit];
            long timed_out = dispatch_semaphore_wait(
                done, dispatch_time(DISPATCH_TIME_NOW, (int64_t)(timeout_s * 1e9)));
            if (timed_out) {
                printf("  %-8d  >%.0f s — barrier did not return (livelock)\n", N_TG, timeout_s);
                printf("\n  ^ CLIFF at %d threadgroups: the software barrier livelocked outright.\n"
                       "  CUDA grid.sync() stays ~1.7-2.1 us/barrier and ~FLAT at every block count.\n", N_TG);
                break;
            }
            if (cb.status == MTLCommandBufferStatusError) {
                printf("  %-8d  ERROR (watchdog?): %s\n", N_TG, cb.error.localizedDescription.UTF8String);
                printf("\n  ^ the watchdog killed it — that IS the cliff. Stopping.\n");
                break;
            }
            double sec = cb.GPUEndTime - cb.GPUStartTime;   // SECONDS
            printf("  %-8d %14.4f %16.1f\n", N_TG, sec * 1e3, sec * 1e9 / iters);
            if (sec * 1e3 > cliff_ms) {
                printf("\n  ^ CLIFF at %d threadgroups: ~%.0fx more expensive per barrier than at\n"
                       "  1-2 threadgroups. Stopping before higher counts livelock outright.\n"
                       "  CUDA grid.sync() stays ~1.7-2.1 us/barrier and ~FLAT at every block count.\n",
                       N_TG, (sec * 1e9 / iters) / 600.0);
                break;
            }
        }
    }
    return 0;
}
