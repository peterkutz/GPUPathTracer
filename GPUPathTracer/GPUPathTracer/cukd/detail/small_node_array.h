// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_SMALL_NODE_ARRAY_H
#define CUKD_SMALL_NODE_ARRAY_H

#include "../utils.h"
#include "node_chunk_array.h"
#include "split_candidate_array.h"
#include <thrust/sequence.h>

namespace cukd {

class SmallNodeArray : private NodeChunkArray {
    friend class KDTree;
    public:
        SmallNodeArray() : complete_(false) {};
        using NodeChunkArray::resize_nodes;
        using NodeChunkArray::resize_elements;
        void resize_element_bits(int nodes);
        void complete();

        using NodeChunkArray::n_nodes;
        using NodeChunkArray::n_elements;

        // Finds the best split candidates and returns indices
        // to the split in SplitCandidateArray as well as the cost, ordered by
        // nodes in active
        void best_split_SAH(cukd::SplitCandidateArray & sca,
                            DevVector<int> & min_sah_split_pos,
                            DevVector<float> & min_sah_cost);
        device::SmallNodeArray dev_array();

    private:
        bool complete_;

        DevVector<int> root_node_idx;
        DevVector<UInt64> element_bits;
        DevVector<UInt64> small_root_node;
};

inline
void
SmallNodeArray::resize_element_bits(int nodes) {
    element_bits.resize(nodes);
    root_node_idx.resize(nodes);
    small_root_node.resize(nodes);
}

inline
device::SmallNodeArray
SmallNodeArray::dev_array() {
    device::SmallNodeArray devarray;
    devarray.na = NodeChunkArray::dev_array();
    devarray.root_node_idx   = root_node_idx.pointer();
    devarray.element_bits    = element_bits.pointer();
    devarray.small_root_node = small_root_node.pointer();
    return devarray;
}

inline
void
SmallNodeArray::complete() {
    complete_ = true;
    resize_element_bits(node_size.size());
    thrust::transform(node_size.begin(), node_size.end(),
                      element_bits.begin(), FillLowestBitsFunctor()); 

    // assign current nodes in split list to root_node_idx
    thrust::sequence(root_node_idx.begin(), root_node_idx.end());
}


} // namespace cukd

#endif  // CUKD_SMALL_NODE_ARRAY_H
