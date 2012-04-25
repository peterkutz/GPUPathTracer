// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_DEV_STRUCTS_H
#define CUKD_DEV_STRUCTS_H

#include "../utils.h"
#include "../primitives.h"

/**********************************************************************************
 *
 * DevNodeChunkArray
 *
 * SoA for the use of NodeChunkArray in custom kernels
 *
 **********************************************************************************/
namespace cukd {
namespace device {

struct NodeArray {
    int* n_elements;
    int* element_idx;
    int* node_size;
    int* node_element_first_idx;

    int* split_axis;
    float* split_position;
    int* left_nodes;
    int* right_nodes;
    int* depth;

    int* n_nodes;
};

struct KDTreeNodeArray {
    NodeArray na;
    int* leaf_idx;
    int* max_depth;
};

struct NodeChunkArray {
    NodeArray na;
    int* node_idx;
    int* chunk_size;
    int* chunk_element_first_idx;
    DevAABBArray node_aabb;
    DevAABBArray chunk_aabb;
    DevAABBArray parent_aabb;
    DevAABBArray triangle_aabb;
};

struct SmallNodeArray {
    NodeChunkArray na;
    int* root_node_idx;
    UInt64* element_bits;
    UInt64* small_root_node;
};

struct SplitCandidateArray {
    int* split_sizes;
    int* first_split_idx;
    float* split_position;
    int* split_direction;
    UInt64* left_elements;
    UInt64* right_elements;
    int* element_idx;
    int* node_element_first_idx;
};

} //  namespace device
} //  namespace cukd
#endif  // CUKD_DEV_STRUCTS_H
