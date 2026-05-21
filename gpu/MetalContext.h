#pragma once
#include <cstdint>

namespace gpu {

// RAII wrapper around MTLDevice + MTLCommandQueue.
// Compiles Metal kernels from embedded MSL source on construction.
// All ObjC types are hidden in Impl — this header is pure C++.
struct MetalContext {
    MetalContext();
    ~MetalContext();

    MetalContext(const MetalContext&) = delete;
    MetalContext& operator=(const MetalContext&) = delete;

    // False if no Metal-capable device was found.
    bool ok() const;

    // Human-readable device name (e.g. "Apple M3 Pro").
    const char* device_name() const;

    // Dispatch a no-op kernel; returns GPU kernel time in microseconds.
    // Prints both GPU kernel time and host-side roundtrip time.
    double noop_roundtrip_us();

    struct Impl;
    Impl* impl_ = nullptr;
};

} // namespace gpu
