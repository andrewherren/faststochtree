#import "MetalContext.h"
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
#include <string>

// ---------------------------------------------------------------------------
// MSL source — compiled at runtime via newLibraryWithSource:.
// Embedding source keeps the build system simple for now; we'll switch to
// offline .metallib compilation once the kernels stabilise.
// ---------------------------------------------------------------------------
static const char* kKernelSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;

// No-op kernel: single thread write to prevent dead-code elimination.
kernel void noop(uint id                  [[thread_position_in_grid]],
                 device volatile uint* out [[buffer(0)]])
{
    if (id == 0) out[0] = id;
}
)MSL";
// ---------------------------------------------------------------------------

namespace gpu {

struct MetalContext::Impl {
    id<MTLDevice>                    device  = nil;
    id<MTLCommandQueue>              queue   = nil;
    id<MTLLibrary>                   library = nil;
    id<MTLComputePipelineState>      noop_pso = nil;
    std::string                      name;
};

MetalContext::MetalContext() : impl_(new Impl()) {
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) {
        fprintf(stderr, "[Metal] No Metal-capable device found.\n");
        return;
    }

    impl_->name  = [[impl_->device name] UTF8String];
    impl_->queue = [impl_->device newCommandQueue];

    // Compile embedded MSL source.
    NSError* err = nil;
    NSString* src = [NSString stringWithUTF8String:kKernelSrc];
    impl_->library = [impl_->device newLibraryWithSource:src options:nil error:&err];
    if (!impl_->library) {
        fprintf(stderr, "[Metal] Shader compile error: %s\n",
                [[err localizedDescription] UTF8String]);
        return;
    }

    id<MTLFunction> fn = [impl_->library newFunctionWithName:@"noop"];
    impl_->noop_pso = [impl_->device newComputePipelineStateWithFunction:fn error:&err];
    if (!impl_->noop_pso) {
        fprintf(stderr, "[Metal] PSO creation error: %s\n",
                [[err localizedDescription] UTF8String]);
    }
}

MetalContext::~MetalContext() { delete impl_; }

bool MetalContext::ok() const {
    return impl_ && impl_->device && impl_->noop_pso;
}

const char* MetalContext::device_name() const {
    return impl_ ? impl_->name.c_str() : "(no device)";
}

double MetalContext::noop_roundtrip_us() {
    if (!ok()) return -1.0;

    // Shared-memory buffer: zero-copy on Apple Silicon.
    id<MTLBuffer> out = [impl_->device
        newBufferWithLength:sizeof(uint32_t)
        options:MTLResourceStorageModeShared];

    auto host_t0 = std::chrono::high_resolution_clock::now();

    id<MTLCommandBuffer>         cmd = [impl_->queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:impl_->noop_pso];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc dispatchThreads:MTLSizeMake(1, 1, 1)
       threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    auto host_t1 = std::chrono::high_resolution_clock::now();

    double gpu_us  = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6;
    double host_us = std::chrono::duration<double, std::micro>(host_t1 - host_t0).count();

    printf("  noop: gpu=%.1f µs  host_roundtrip=%.1f µs\n", gpu_us, host_us);
    return gpu_us;
}

} // namespace gpu
