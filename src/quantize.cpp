#include "faststochtree/quantize.hpp"
#include <algorithm>

namespace bart {

static int map_to_cut(float v, const std::vector<float>& cuts) {
    // Index of the largest cut-point <= v
    auto it = std::upper_bound(cuts.begin(), cuts.end(), v);
    int idx = (int)(it - cuts.begin()) - 1;
    if (idx < 0) idx = 0;
    if (idx >= (int)cuts.size()) idx = (int)cuts.size() - 1;
    return idx;
}

QuantizedX quantize(const float* X, int n, int p, int max_cuts) {
    QuantizedX qx;
    qx.n = n; qx.p = p;
    qx.data.resize(n * p);
    qx.cuts.resize(p);

    for (int j = 0; j < p; j++) {
        // Collect and sort unique values for feature j
        std::vector<float> vals(n);
        for (int i = 0; i < n; i++) vals[i] = X[i * p + j];
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        // Select cut-points: evenly-spaced quantiles if too many unique values
        auto& cuts = qx.cuts[j];
        if ((int)vals.size() <= max_cuts) {
            cuts = vals;
        } else {
            cuts.resize(max_cuts);
            for (int k = 0; k < max_cuts; k++) {
                int idx = (int)((k + 0.5f) * vals.size() / max_cuts);
                cuts[k] = vals[idx];
            }
        }

        // Map each observation to its cut-point index; store column-major
        for (int i = 0; i < n; i++)
            qx.data[j * n + i] = (uint8_t)map_to_cut(X[i * p + j], cuts);
    }
    return qx;
}

QuantizedX quantize_with_cuts(const float* X, int n, const QuantizedX& ref) {
    QuantizedX qx;
    qx.n = n; qx.p = ref.p;
    qx.cuts = ref.cuts;
    qx.data.resize(n * ref.p);

    for (int j = 0; j < ref.p; j++)
        for (int i = 0; i < n; i++)
            qx.data[j * n + i] = (uint8_t)map_to_cut(X[i * ref.p + j], ref.cuts[j]);

    return qx;
}

} // namespace bart
