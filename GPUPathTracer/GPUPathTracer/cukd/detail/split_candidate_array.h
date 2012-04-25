// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_SPLIT_CANDIDATE_ARRAY_H
#define CUKD_SPLIT_CANDIDATE_ARRAY_H

#include "node_chunk_array.h"

namespace cukd {

class SplitCandidateArray {
    public:
        SplitCandidateArray() {};

        void split_candidates(NodeChunkArray & nca);
        void resize(int n_nodes, int n_elements);
        device::SplitCandidateArray dev_array();

    private:
        DevVector<int> split_sizes;
        DevVector<int> split_direction;
        DevVector<int> first_split_idx;
        DevVector<float> split_position;
        DevVector<UInt64> left_elements;
        DevVector<UInt64> right_elements;
        DevVector<int> root_idx;
        DevVector<int> element_idx;
        DevVector<int> first_element_idx;
};

inline void
SplitCandidateArray::resize(int n_nodes, int n_elements) {
    split_sizes.resize(n_nodes);
    first_split_idx.resize(n_nodes);
    split_position.resize(6*n_elements);
    split_direction.resize(6*n_elements);
    left_elements.resize(6*n_elements);
    right_elements.resize(6*n_elements);
    element_idx.resize(n_elements);
    first_element_idx.resize(n_nodes);
}

inline device::SplitCandidateArray
SplitCandidateArray::dev_array() {
    device::SplitCandidateArray res;
    res.split_sizes    =  split_sizes.pointer();
    res.first_split_idx=  first_split_idx.pointer();
    res.split_position =  split_position.pointer();
    res.split_direction =  split_direction.pointer();
    res.left_elements  =  left_elements.pointer();
    res.right_elements =  right_elements.pointer();
    res.element_idx = element_idx.pointer();
    res.node_element_first_idx = first_element_idx.pointer();
    return res;
}

}

#endif
