// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include "kdtree_node_array.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

namespace cukd {

// Kernel declarations
namespace device {
    __global__
    void
    tree_update_children_small_kernel(device::KDTreeNodeArray tree,
            int n_nodes);

    __global__
    void
    update_tree_children_from_small_kernel(int n_active_nodes, int n_small,
                                           device::KDTreeNodeArray tree, int* tags,
                                           int* small_offset, int* active_indices);
    __global__
    void
    leaf_elements_kernel(device::SmallNodeArray active, device::SplitCandidateArray sca,
                         int old_small_nodes, int* marks, int* elem_offsets, int* result);
}  // namespace device

void
KDTreeNodeArray::print() {
    left_nodes.print("kdtree_node_array::left_nodes");
    right_nodes.print("KDTreeNodeArray::right_nodes");
    split_axis.print("KDTreeNodeArray::split_axis");
    split_position.print("KDTreeNodeArray::split_position");
    depth.print("KDTreeNodeArray::depth");
    leaf_idx.print("KDTreeNodeArray::leaf_index");
    node_size.print("KDTreeNodeArray::leaf_size");
    node_element_first_idx.print("KDTreeNodeArray::leaf_first_elem");
    element_idx.print("KDTreeNodeArray::element_idx");
}

std::pair<int, int>
KDTreeNodeArray::update_leaves(SmallNodeArray & small_nca,
                               cukd::SplitCandidateArray & sca,
                               DevVector<int> & new_elements,
                               DevVector<int> & marks,
                               DevVector<int> & mark_offsets) {
    int n_nodes_old = n_nodes();
    int n_elements_old = n_elements();
    int n_leaves_old = n_leaves();
    int small_nodes = small_nca.n_nodes();

    DevVector<int> leaf_elements, elem_offsets, leaf_element_offsets;
    DevVector<int> leaf_element_sizes;

    int new_leaf_nodes = mark_offsets.get_at(mark_offsets.size() - 1);
    leaf_element_sizes.resize(new_leaf_nodes);
    leaf_element_offsets.resize(new_leaf_nodes);
    elem_offsets.resize(small_nodes);

    thrust::copy_if(new_elements.begin(), new_elements.end(),
            marks.begin(), leaf_element_sizes.begin(), GreaterThanZero());

    int new_leaf_elements = thrust::reduce(new_elements.begin(), new_elements.end());

    thrust::exclusive_scan(leaf_element_sizes.begin(), leaf_element_sizes.end(),
                           leaf_element_offsets.begin());
    thrust::exclusive_scan(new_elements.begin(), new_elements.end(), elem_offsets.begin());

    leaf_elements.resize(new_leaf_elements);

    get_leaf_elements(small_nca, sca, small_nodes, marks, elem_offsets, leaf_elements);

    resize_nodes(n_nodes_old + small_nodes);
    resize_elements(n_elements_old + new_leaf_elements);
    resize_leaves(n_leaves_old + new_leaf_nodes);

    thrust::copy(leaf_element_sizes.begin(), leaf_element_sizes.end(),
                 node_size.begin() + n_leaves_old);

    int next_off = 0;
    if(n_leaves_old != 0) {
        next_off =  node_element_first_idx.get_at(n_leaves_old - 1)
                    + node_size.get_at(n_leaves_old - 1);
        thrust::transform(leaf_element_offsets.begin(), leaf_element_offsets.end(),
                     thrust::constant_iterator<int>(next_off),
                     node_element_first_idx.begin() + n_leaves_old,
                     thrust::plus<int>());
    } else {
        thrust::copy(leaf_element_offsets.begin(), leaf_element_offsets.end(),
                     node_element_first_idx.begin() + n_leaves_old);
    }

    thrust::copy(leaf_elements.begin(), leaf_elements.end(),
                 element_idx.begin() + next_off);

    return std::make_pair(n_leaves_old, new_leaf_nodes);
}

void
KDTreeNodeArray::update_children_small() {
    dim3 grid(IntegerDivide(256)(n_nodes()), 1, 1);
    dim3 blocks(256,1,1);
    device::tree_update_children_small_kernel<<<grid, blocks>>>(dev_array(), n_nodes());
    CUT_CHECK_ERROR("tree_update_children_small_kernel failed");
}

void
KDTreeNodeArray::update_tree_children_from_small(int n_nodes_active, int n_nodes_small,
                                                 DevVector<int> & small_tags,
                                                 DevVector<int> & child_diff,
                                                 DevVector<int> & active_indices) {
    dim3 grid(IntegerDivide(256)(n_nodes_active),1,1);
    dim3 blocks(256,1,1);
    device::update_tree_children_from_small_kernel<<<grid,blocks>>>(
            n_nodes_active, n_nodes_small, dev_array(),
            small_tags.pointer(), child_diff.pointer(),
            active_indices.pointer());
    CUT_CHECK_ERROR("update_tree_children_from_small_kernel failed");
}

void 
KDTreeNodeArray::get_leaf_elements(cukd::SmallNodeArray & active, 
                   cukd::SplitCandidateArray & sca,
                   int old_small_nodes, DevVector<int> & marks,
                   DevVector<int> & elem_offsets, DevVector<int> & result) {
    dim3 grid(IntegerDivide(256)(old_small_nodes),1,1);
    dim3 blocks(256,1,1);
    device::leaf_elements_kernel<<<grid, blocks>>>(active.dev_array(), sca.dev_array(),
                                        old_small_nodes, marks.pointer(),
                                       elem_offsets.pointer(), result.pointer());
}

}  // namespace cukd
