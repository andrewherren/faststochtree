#pragma once
#include <cstdint>
#include <vector>

namespace bart {

struct QuantizedX {
    int n, p;
    std::vector<uint8_t>            data;  // row-major [n*p]: data[i*p + j]
    std::vector<std::vector<float>> cuts;  // cuts[j]: sorted cut-point float values

    uint8_t at(int i, int j) const { return data[j * n + i]; }
};

// Quantize float X[n*p] (row-major) using up to max_cuts cut-points per feature.
QuantizedX quantize(const float* X, int n, int p, int max_cuts = 255);

// Quantize new observations using the cut-point tables from a reference QuantizedX.
// Used to map test data to the same indices as training.
QuantizedX quantize_with_cuts(const float* X, int n, const QuantizedX& ref);

} // namespace bart
