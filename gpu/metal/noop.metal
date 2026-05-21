// Placeholder — compiled from the embedded string in MetalContext.mm for now.
// This file documents the kernel interface; we'll switch to offline .metallib
// compilation once the histogram kernel is ready.
//
// #include <metal_stdlib>
// using namespace metal;
//
// kernel void noop(uint id                   [[thread_position_in_grid]],
//                  device volatile uint* out  [[buffer(0)]])
// { if (id == 0) out[0] = id; }
