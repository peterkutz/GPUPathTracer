// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_KDTREE_NODE_ARRAY_H
#define CUKD_KDTREE_NODE_ARRAY_H

#include "../utils.h"
#include "dev_structs.h"
#include "node_chunk_array.h"
#include "node_array.h"
#include "small_node_array.h"

namespace cukd {

class SplitCandidateArray;

class KDTreeNodeArray : private NodeArray {
    friend class KDTree;
    public:
        KDTreeNodeArray() : n_leaves_(0) {
            _max_depth.set(0);
        };

        void resize_nodes(int nodes);
        using NodeArray::resize_elements;
        void resize_leaves(int leaves);

        using NodeArray::n_nodes;
        using NodeArray::n_elements;
        int n_leaves() const { return n_leaves_; };
        int max_depth() const { return _max_depth.get(); };

        void clear();

        std::pair<int,int> update_leaves(SmallNodeArray & small_nca,
                                         cukd::SplitCandidateArray & sca,
                                         DevVector<int> & n_elements,
                                         DevVector<int> & marks,
                                         DevVector<int> & mark_offsets);
        // Sets negative indices to small root nodes to the correct indices
        // TODO: change after small node stage implementation
        void update_children_small();

        // Updates the Tree's children after small nodes have been removed
        // from the next list. This includes indices to already expected
        // median split as well as indices to the small list's roots, which
        // are ste to be negative for a fast final tree update once the small
        // node stage is completed.
        void update_tree_children_from_small(int n_nodes_active, int n_nodes_small,
                                             DevVector<int> & small_tags,
                                             DevVector<int> & child_diff,
                                             DevVector<int> & active_indices);
        // updates the tree with elements after small node stage
        void get_leaf_elements(cukd::SmallNodeArray & active, cukd::SplitCandidateArray & sca,
                           int old_small_nodes, DevVector<int> & marks,
                           DevVector<int> & elem_offsets, DevVector<int> & result);

        void print();

        device::KDTreeNodeArray dev_array();

    private:
        int n_leaves_;
        DevVariable<int> _max_depth;
        DevVector<int> leaf_idx;
};

inline
void
KDTreeNodeArray::clear() {
    n_leaves_  = 0;
    leaf_idx.clear();
    NodeArray::clear();
};

inline
void
KDTreeNodeArray::resize_nodes(int nodes) {
    n_nodes_ = nodes;
    split_axis.resize(nodes);
    split_position.resize(nodes);
    left_nodes.resize(nodes);
    right_nodes.resize(nodes);
    depth.resize(nodes);
    leaf_idx.resize(nodes,-1);
};

inline
void
KDTreeNodeArray::resize_leaves(int leaves) {
    n_leaves_ = leaves;
    node_size.resize(leaves);
    node_element_first_idx.resize(leaves);
};

inline
device::KDTreeNodeArray
KDTreeNodeArray::dev_array() {
    device::KDTreeNodeArray devarray;
    devarray.na = NodeArray::dev_array();
    devarray.max_depth = _max_depth.pointer();
    devarray.leaf_idx = leaf_idx.pointer();
    return devarray;
}

}  // namespace cukd
#endif  // CUKD_KDTREE_NODE_ARRAY_H
