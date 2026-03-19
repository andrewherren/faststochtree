#pragma once
#include <cstdint>
#include <vector>
#include <cassert>

// v3-quantized: threshold is now uint8_t (quantized cut-point index).

namespace bart {

struct Node {
    int     split_var;  // feature index; -1 = leaf
    uint8_t threshold;  // quantized cut-point index
    float   value;      // leaf prediction (used only when split_var == -1)
    int     depth;
    int     left;       // index into Tree::nodes; -1 if leaf
    int     right;
};

struct Tree {
    std::vector<Node> nodes;  // nodes[0] is always the root

    Tree() {
        nodes.push_back({-1, 0, 0.f, 0, -1, -1});  // root
    }

    bool is_leaf(int k) const { return nodes[k].split_var == -1; }

    // Traverse observation i using row-major quantized X (Xq[i*p + j]); returns node index.
    int traverse(const uint8_t* Xq, int i, int p) const {
        int k = 0;
        while (!is_leaf(k)) {
            uint8_t val = Xq[i * p + nodes[k].split_var];
            k = (val <= nodes[k].threshold) ? nodes[k].left : nodes[k].right;
        }
        return k;
    }

    // All leaf node indices
    std::vector<int> leaves() const {
        std::vector<int> r;
        collect_leaves(0, r);
        return r;
    }

    // All internal node indices whose two children are both leaves
    std::vector<int> leaf_parents() const {
        std::vector<int> r;
        collect_leaf_parents(0, r);
        return r;
    }

    // Grow leaf k into an internal node; appends two new leaf children
    void grow(int k, int var, uint8_t threshold) {
        assert(is_leaf(k));
        int li = (int)nodes.size();
        nodes.push_back({-1, 0, nodes[k].value, nodes[k].depth + 1, -1, -1});
        int ri = (int)nodes.size();
        nodes.push_back({-1, 0, nodes[k].value, nodes[k].depth + 1, -1, -1});
        // Must index nodes[k] after push_back in case of reallocation
        nodes[k].split_var  = var;
        nodes[k].threshold  = threshold;
        nodes[k].left       = li;
        nodes[k].right      = ri;
    }

    // Prune leaf-parent k back to a leaf (children become orphaned slots)
    void prune(int k) {
        assert(!is_leaf(k));
        assert(is_leaf(nodes[k].left) && is_leaf(nodes[k].right));
        nodes[k].split_var = -1;
        nodes[k].left      = -1;
        nodes[k].right     = -1;
    }

private:
    void collect_leaves(int k, std::vector<int>& r) const {
        if (is_leaf(k)) { r.push_back(k); return; }
        collect_leaves(nodes[k].left,  r);
        collect_leaves(nodes[k].right, r);
    }
    void collect_leaf_parents(int k, std::vector<int>& r) const {
        if (is_leaf(k)) return;
        if (is_leaf(nodes[k].left) && is_leaf(nodes[k].right))
            r.push_back(k);
        collect_leaf_parents(nodes[k].left,  r);
        collect_leaf_parents(nodes[k].right, r);
    }
};

} // namespace bart
