#pragma once
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

// v5-fixed-depth: implicit binary heap tree.
//
// 1-indexed nodes. With depth=6:
//   half_size = 63  (last internal node index)
//   full_size = 127 (last node index; leaf_value defined for [0..127])
//   Internal nodes: [1, 63]  — split (split_var>0) or real leaf (split_var==-1)
//   Always-leaf:    [64, 127] — bottom level, never split
//
// split_var sentinel encoding:
//   -1  real unsplit leaf (reachable, no split)
//    0  phantom slot (unreachable child of a leaf; never written after init)
//   >0  split on variable split_var[k]-1
//
// This lets leaves(out) use a plain linear scan — phantoms (split_var==0) are
// excluded by is_leaf(), no parent-check needed.

namespace bart {

struct Tree {
    int depth;      // max leaf depth
    int half_size;  // (1<<depth)-1: last internal node index
    int full_size;  // (1<<(depth+1))-1: last node index

    std::vector<int>     split_var;   // [0..half_size]; -1=real leaf, 0=phantom, >0=split on var split_var[k]-1
    std::vector<uint8_t> threshold;   // [0..half_size]
    std::vector<float>   leaf_value;  // [0..full_size]; valid for any node index

    explicit Tree(int depth_ = 6)
        : depth(depth_),
          half_size((1 << depth_) - 1),
          full_size((1 << (depth_ + 1)) - 1),
          split_var(half_size + 1, 0),
          threshold(half_size + 1, 0),
          leaf_value(full_size + 1, 0.f) { split_var[1] = -1; }

    bool is_leaf(int k) const { return k > half_size || split_var[k] < 0; }

    // Depth of node k in the heap (root = 0).
    static int depth_of(int k) {
        int d = 0;
        while (k > 1) { k >>= 1; d++; }
        return d;
    }

    // Traverse observation i (column-major Xq[j*n+i]); returns leaf node index.
    int traverse(const uint8_t* Xq, int i, int n) const {
        int k = 1;
        while (k <= half_size && split_var[k] > 0) {
            int var = split_var[k] - 1;
            k = 2*k + (Xq[var * n + i] > threshold[k] ? 1 : 0);
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
        // Children at 2*k > half_size are always-leaf (caught by is_leaf's first check);
        // only mark internal children as real leaves.
        if (2*k <= half_size) {
            split_var[2*k]   = -1;
            split_var[2*k+1] = -1;
        }
    }

    // Prune internal node k (both children must be leaves).
    void prune(int k) {
        assert(!is_leaf(k));
        assert(is_leaf(2*k) && is_leaf(2*k+1));
        split_var[k] = -1;  // node becomes a real leaf
        if (2*k <= half_size) {
            split_var[2*k]   = 0;  // children become phantom
            split_var[2*k+1] = 0;
        }
    }

    // Reset to a single-leaf root (used by GFR).
    void reset() {
        std::fill(split_var.begin(), split_var.end(), 0);
        split_var[1] = -1;
    }

    std::vector<int> leaves() const {
        std::vector<int> r;
        for (int k = 1; k <= full_size; k++)
            if (is_leaf(k)) r.push_back(k);
        return r;
    }

    std::vector<int> leaf_parents() const {
        std::vector<int> r;
        for (int k = 1; k <= half_size; k++)
            if (split_var[k] > 0 && is_leaf(2*k) && is_leaf(2*k+1)) r.push_back(k);
        return r;
    }
};

} // namespace bart
