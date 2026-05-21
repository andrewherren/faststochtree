#include "MetalContext.h"
#include <cstdio>

int main() {
    printf("=== faststochtree GPU bench ===\n\n");

    gpu::MetalContext ctx;
    if (!ctx.ok()) {
        fprintf(stderr, "Metal not available — cannot run GPU bench.\n");
        return 1;
    }
    printf("Device: %s\n\n", ctx.device_name());

    // First dispatch includes runtime shader compilation + pipeline cache population.
    printf("--- warmup (includes shader compile) ---\n");
    ctx.noop_roundtrip_us();

    printf("\n--- noop kernel roundtrip (5 runs) ---\n");
    for (int i = 0; i < 5; ++i)
        ctx.noop_roundtrip_us();

    return 0;
}
