#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

// v5-fixed-depth: implicit binary heap tree.
//
// 1-indexed nodes. With depth=6:
//   half_size = 63  (last internal node index; split_var defined for [0..63])
//   full_size = 127 (last node index; leaf_value defined for [0..127])
//   Internal nodes: [1, 63]  — can be split or unsplit (leaf)
//   Always-leaf:    [64, 127] — bottom level, never split
//
// This gives 64 × (4+1) = 320 bytes for split_var + threshold, matching the
// plan's performance analysis.

namespace bart {

struct Tree {
    int depth;      // max leaf depth
    int half_size;  // (1<<depth)-1: last internal node index
    int full_size;  // (1<<(depth+1))-1: last node index

    std::vector<int>     split_var;   // [0..half_size]; 0 = unsplit (leaf); actual var = split_var[k]-1
    std::vector<uint8_t> threshold;   // [0..half_size]
    std::vector<float>   leaf_value;  // [0..full_size]; valid for any node index

    explicit Tree(int depth_ = 6)
        : depth(depth_),
          half_size((1 << depth_) - 1),
          full_size((1 << (depth_ + 1)) - 1),
          // Size all three arrays to full_size+1 so the branchless traverse
          // can safely read split_var[k] for any k in [1, full_size].
          // Always-leaf slots (k > half_size) stay 0 and never get written.
          split_var(full_size + 1, 0),
          threshold(full_size + 1, 0),
          leaf_value(full_size + 1, 0.f) {}

    bool is_leaf(int k) const { return k > half_size || split_var[k] == 0; }

    // Depth of node k in the heap (root = 0).
    static int depth_of(int k) {
        int d = 0;
        while (k > 1) { k >>= 1; d++; }
        return d;
    }

    // Traverse observation i (column-major Xq[j*n+i]); returns leaf node index.
    // Branchless fixed-iteration form: compiles to cmov, no data-dependent branches.
    // var = (split_var[k]-1)*is_split keeps the feature index 0 when not splitting,
    // ensuring the Xq load always reads a valid column address.
    int traverse(const uint8_t* Xq, int i, int n) const {
        int k = 1;
        for (int d = 0; d < depth; d++) {
            int is_split = (k <= half_size) & (split_var[k] != 0);
            int var      = (split_var[k] - 1) * is_split;  // 0 if not split
            int go_right = is_split & (Xq[var * n + i] > threshold[k]);
            k            = is_split ? (2 * k + go_right) : k;
        }
        return k;
    }

    // Grow leaf k: make it an internal node, push leaf value to children.
    void grow(int k, int var, uint8_t thresh) {
        assert(is_leaf(k) && k <= half_size);
        split_var[k]      = var + 1;
        threshold[k]      = thresh;
        leaf_value[2*k]   = leaf_value[k];
        leaf_value[2*k+1] = leaf_value[k];
    }

    // Prune internal node k (both children must be leaves).
    void prune(int k) {
        assert(!is_leaf(k));
        assert(is_leaf(2*k) && is_leaf(2*k+1));
        split_var[k] = 0;
    }

    // Reset to a single-leaf root (used by GFR).
    void reset() { std::fill(split_var.begin(), split_var.end(), 0); }

    std::vector<int> leaves() const {
        std::vector<int> r;
        leaves(r);
        return r;
    }

    std::vector<int> leaf_parents() const {
        std::vector<int> r;
        leaf_parents(r);
        return r;
    }

    // Out-parameter variants — write into pre-allocated buffer (no heap alloc).
    // Parent-check scan: a phantom slot's parent is also a leaf, so excluding
    // any k where is_leaf(k>>1) is true yields only reachable leaves — no DFS
    // stack needed.
    void leaves(std::vector<int>& out) const {
        out.clear();
        for (int k = 1; k <= full_size; k++)
            if (is_leaf(k) && (k == 1 || !is_leaf(k >> 1))) out.push_back(k);
    }

    void leaf_parents(std::vector<int>& out) const {
        out.clear();
        for (int k = 1; k <= half_size; k++)
            if (!is_leaf(k) && is_leaf(2*k) && is_leaf(2*k+1)) out.push_back(k);
    }
};

} // namespace bart
