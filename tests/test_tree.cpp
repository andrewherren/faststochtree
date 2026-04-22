#include "faststochtree/tree.hpp"
#include <gtest/gtest.h>
#include <vector>

using bart::Tree;

// ── sentinel encoding invariants ──────────────────────────────────────────────
//   split_var[k] == -1  →  real leaf (reachable, no split)
//   split_var[k] ==  0  →  phantom  (unreachable, child of a leaf)
//   split_var[k]  >  0  →  internal split on variable split_var[k]-1

TEST(TreeSentinel, FreshTreeRootIsOnlyLeaf) {
    Tree t(3);  // half_size=7, full_size=15
    auto ls = t.leaves();
    ASSERT_EQ(ls.size(), 1u);
    EXPECT_EQ(ls[0], 1);
}

// Regression: internal phantom nodes (split_var==0, k <= half_size) must not
// appear in leaves(). Before the sentinel fix they did because the old is_leaf
// checked split_var[k] == 0 (no split) which matched phantoms too.
TEST(TreeSentinel, LeavesExcludesPhantomInternalNodes) {
    Tree t(3);
    t.grow(1, 0, 128);  // root → children 2,3; nodes 4-7 become phantom internal
    auto ls = t.leaves();
    ASSERT_EQ(ls.size(), 2u);
    EXPECT_EQ(ls[0], 2);
    EXPECT_EQ(ls[1], 3);
    for (int k : ls)
        EXPECT_EQ(t.split_var[k], -1) << "leaf k=" << k << " must have split_var==-1";
}

// Regression: always-leaf phantom nodes (k > half_size, split_var==0) must not
// appear in leaves(). The buggy is_leaf used `k > half_size || split_var[k] < 0`,
// which included all k > half_size unconditionally regardless of split_var.
TEST(TreeSentinel, LeavesExcludesPhantomAlwaysLeafNodes) {
    Tree t(3);  // always-leaf range: [8..15]
    t.grow(1, 0, 128);  // leaves at depth 1 only; no always-leaf node is reachable
    auto ls = t.leaves();
    for (int k : ls)
        EXPECT_LE(k, t.half_size) << "phantom always-leaf k=" << k << " must not appear";
    EXPECT_EQ(ls.size(), 2u);
}

// When all always-leaf nodes are genuinely reachable (fully grown tree), they
// must ALL appear in leaves().
TEST(TreeSentinel, RealAlwaysLeafNodesIncluded) {
    Tree t(2);  // half_size=3, full_size=7; always-leaf range: [4..7]
    t.grow(1, 0, 128);
    t.grow(2, 0, 64);
    t.grow(3, 0, 192);
    auto ls = t.leaves();
    ASSERT_EQ(ls.size(), 4u);
    for (int k : ls)
        EXPECT_GT(k, t.half_size) << "all leaves should be in the always-leaf range";
}

// Partial growth: some always-leaf nodes real, others phantom.
TEST(TreeSentinel, PartialGrowthMixedAlwaysLeaf) {
    Tree t(2);  // always-leaf: [4..7]
    t.grow(1, 0, 128);  // → {2,3}
    t.grow(2, 0, 64);   // → {4,5} real always-leaf; 3 remains real internal leaf
                        //   nodes 6,7 are phantom always-leaf (parent 3 not grown)
    auto ls = t.leaves();
    ASSERT_EQ(ls.size(), 3u);
    EXPECT_EQ(ls[0], 3);
    EXPECT_EQ(ls[1], 4);
    EXPECT_EQ(ls[2], 5);
    for (int k : ls)
        EXPECT_NE(k, 6) << "phantom always-leaf 6 must not appear";
    for (int k : ls)
        EXPECT_NE(k, 7) << "phantom always-leaf 7 must not appear";
}

// Regression: leaf_parents() must not return phantom nodes as prune candidates.
// Before the sentinel fix, phantoms with split_var==0 passed !is_leaf(k) since
// is_leaf checked < 0, making 0 look like a non-leaf (split node).
TEST(TreeSentinel, LeafParentsExcludesPhantomNodes) {
    Tree t(3);
    t.grow(1, 0, 128);  // root → {2,3}: node 1 is the only valid prune candidate
    auto lp = t.leaf_parents();
    ASSERT_EQ(lp.size(), 1u);
    EXPECT_EQ(lp[0], 1);
    for (int k : lp)
        EXPECT_GT(t.split_var[k], 0) << "prune candidate k=" << k << " must have split_var>0";
}

// leaf_parents() only includes a node when BOTH children are real leaves.
TEST(TreeSentinel, LeafParentsRequiresBothChildrenLeaf) {
    Tree t(3);
    t.grow(1, 0, 128);  // root → {2,3}
    t.grow(2, 0, 64);   // node 2 → {4,5}; now node 1 has one internal child
    auto lp = t.leaf_parents();
    ASSERT_EQ(lp.size(), 1u);
    EXPECT_EQ(lp[0], 2);  // node 1 is NOT a valid prune candidate (child 2 is internal)
}

// grow() sets correct sentinel values, including for always-leaf children.
TEST(TreeSentinel, GrowSentinelValues) {
    Tree t(2);  // half_size=3
    t.grow(1, 0, 128);  // children 2,3 are internal range (≤ half_size)
    EXPECT_EQ(t.split_var[1],  1);   // split on var 0
    EXPECT_EQ(t.split_var[2], -1);   // real leaf
    EXPECT_EQ(t.split_var[3], -1);   // real leaf

    t.grow(2, 0, 64);   // children 4,5 are always-leaf (> half_size)
    EXPECT_EQ(t.split_var[2],  1);   // now a split
    EXPECT_EQ(t.split_var[4], -1);   // real always-leaf
    EXPECT_EQ(t.split_var[5], -1);   // real always-leaf
    EXPECT_EQ(t.split_var[6],  0);   // phantom always-leaf (parent 3 not grown)
    EXPECT_EQ(t.split_var[7],  0);   // phantom always-leaf
}

// prune() restores correct sentinel values.
TEST(TreeSentinel, PruneSentinelValues) {
    Tree t(2);
    t.grow(1, 0, 128);
    t.grow(2, 0, 64);
    t.prune(2);   // children 4,5 (always-leaf) become phantom
    EXPECT_EQ(t.split_var[2], -1);   // node 2 back to real leaf
    EXPECT_EQ(t.split_var[4],  0);   // phantom
    EXPECT_EQ(t.split_var[5],  0);   // phantom

    t.prune(1);   // children 2,3 become phantom
    EXPECT_EQ(t.split_var[1], -1);
    EXPECT_EQ(t.split_var[2],  0);
    EXPECT_EQ(t.split_var[3],  0);
}

// is_leaf() must correctly classify all three sentinel categories including
// the always-leaf range.
TEST(TreeSentinel, IsLeafAllCategories) {
    Tree t(2);  // half_size=3, full_size=7
    t.grow(1, 0, 128);  // → {2,3}
    t.grow(2, 0, 64);   // → {4,5} real always-leaf; 6,7 phantom always-leaf

    EXPECT_FALSE(t.is_leaf(1));  // internal split
    EXPECT_FALSE(t.is_leaf(2));  // internal split
    EXPECT_TRUE (t.is_leaf(3));  // real internal leaf (split_var==-1)
    EXPECT_TRUE (t.is_leaf(4));  // real always-leaf  (split_var==-1)
    EXPECT_TRUE (t.is_leaf(5));  // real always-leaf  (split_var==-1)
    EXPECT_FALSE(t.is_leaf(6));  // phantom always-leaf (split_var==0)
    EXPECT_FALSE(t.is_leaf(7));  // phantom always-leaf (split_var==0)
}

// reset() returns the tree to a clean single-leaf root with no phantom leakage.
TEST(TreeSentinel, ResetRestoresCleanState) {
    Tree t(3);
    t.grow(1, 0, 128);
    t.grow(2, 0, 64);
    t.reset();

    EXPECT_EQ(t.split_var[1], -1);
    for (int k = 2; k <= t.full_size; k++)
        EXPECT_EQ(t.split_var[k], 0) << "k=" << k << " must be phantom after reset";

    auto ls = t.leaves();
    ASSERT_EQ(ls.size(), 1u);
    EXPECT_EQ(ls[0], 1);
}
