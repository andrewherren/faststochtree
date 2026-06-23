// dispatch_overhead.mm — how expensive is one Metal compute dispatch?
//
// Claim demonstrated: Metal has a high *per-launch* cost and no way to amortize a
// long sequence of tiny dependent kernels. The BART sampler updates the residual at
// least once per tree, and the initial Metal experiments used several phases (each
// needing a device-wide sync) per tree — so over a run of T trees x many sweeps, a
// dispatch happens a great many times. On Metal each phase that must observe a
// device-wide sync is its own command buffer at ~tens of microseconds to encode +
// commit, and that constant is multiplied by all those phases. CUDA Graphs replay the
// same sequence for ~1.6 us/launch (T4) — Metal has no equivalent (see ../cuda/reprex.ipynb).
//
// We measure an empty ("noop") kernel two ways:
//   (a) N dispatches encoded into ONE command buffer  -> GPU-side cost / N
//       (this is the cheap case, but it CANNOT cross a device-wide barrier, which is
//        exactly what every BART phase boundary needs — so it is not usable here)
//   (b) N SEPARATE command buffers, encode+commit+wait -> host wall-clock / N
//       (this is the realistic per-phase cost when phases must synchronize)
//
// Build: make dispatch_overhead    Run: ./dispatch_overhead
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdio>

static const char* kSrc = R"(
#include <metal_stdlib>
using namespace metal;
kernel void noop(uint tid [[thread_position_in_grid]]) { }
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
        id<MTLFunction> fn = [lib newFunctionWithName:@"noop"];
        id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!pso) { printf("pipeline failed: %s\n", err.localizedDescription.UTF8String); return 1; }

        const int N = 2000;
        const MTLSize one = MTLSizeMake(1, 1, 1);

        // (a) one command buffer, N encoded dispatches.
        {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:pso];
            for (int i = 0; i < N; ++i)
                [e dispatchThreadgroups:one threadsPerThreadgroup:one];
            [e endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            double sec = cb.GPUEndTime - cb.GPUStartTime;   // NOTE: these are SECONDS
            printf("  (a) one cmd buffer, %d encoded dispatches : %7.3f us/dispatch (GPU)\n",
                   N, sec * 1e6 / N);
        }

        // (b) N separate command buffers (the realistic per-sync-phase cost).
        {
            auto t0 = std::chrono::steady_clock::now();
            id<MTLCommandBuffer> last = nil;
            for (int i = 0; i < N; ++i) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
                [e setComputePipelineState:pso];
                [e dispatchThreadgroups:one threadsPerThreadgroup:one];
                [e endEncoding];
                [cb commit];
                last = cb;
            }
            [last waitUntilCompleted];
            auto t1 = std::chrono::steady_clock::now();
            double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
            printf("  (b) %d separate command buffers           : %7.3f us/dispatch (host: encode+commit)\n",
                   N, us);
        }

        printf("\n  (b) is the figure that matters: ~tens of us per phase, and a phase recurs\n"
               "  several times per tree, every tree, every sweep — with no CUDA-Graph-style\n"
               "  replay to amortize it. The per-dispatch constant is what adds up.\n");
    }
    return 0;
}
