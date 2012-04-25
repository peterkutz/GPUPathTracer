// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include "kdtree_node_array.h"

namespace cukd {
namespace device {

__global__
void
leaf_elements_kernel(device::SmallNodeArray active, device::SplitCandidateArray sca,
                     int old_small_nodes, int* marks, int* elem_offsets, int* result) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    int root_node;
    int i, first_element;
    int count = 0;
    UInt64 mask, element_bits;
    if(tid < old_small_nodes) {
        if(marks[tid] == 1) {
            root_node = active.root_node_idx[tid];
            element_bits = active.element_bits[tid];
            first_element = sca.node_element_first_idx[root_node];
            count = 0;
            for(i = 0; i < 64; ++i) {
                mask = (UInt64) 1 << i;
                if(mask & element_bits) {
                    result[elem_offsets[tid] + count] = sca.element_idx[first_element + i];
                    count++;
                }
            }
        }
    }
}

__global__
void
update_tree_children_from_small_kernel(int n_active_nodes, int n_small,
                                       device::KDTreeNodeArray tree, int* tags,
                                       int* small_offset, int* active_indices) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;

    if(tid < n_active_nodes) {
        // determine the correct node to change
        int tree_idx = active_indices[tid];
        int left_idx = tid;
        int right_idx = tid + n_active_nodes;

        tree.na.left_nodes[tree_idx] = tags[left_idx]*(-n_small - small_offset[left_idx])
                                     + (1 - tags[left_idx])*(tree.na.left_nodes[tree_idx] -
                                                             small_offset[left_idx]);

        tree.na.right_nodes[tree_idx] = tags[right_idx]*(-n_small - small_offset[right_idx])
                                     + (1 - tags[right_idx])*(tree.na.right_nodes[tree_idx] -
                                                              small_offset[right_idx]);
    }
}

__global__
void
tree_update_children_small_kernel(device::KDTreeNodeArray tree, int
        n_nodes) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid < n_nodes) {
        if(tree.na.left_nodes[tid] < 0) {
            tree.na.left_nodes[tid] = n_nodes - tree.na.left_nodes[tid] - 1;
        }
        if(tree.na.right_nodes[tid] < 0) {
            tree.na.right_nodes[tid] = n_nodes - tree.na.right_nodes[tid] - 1;
        }
    }
}

}  // namespace device
}  // namespace cukd 
