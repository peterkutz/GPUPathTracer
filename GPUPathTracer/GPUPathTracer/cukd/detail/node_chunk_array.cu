// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include "node_chunk_array.h"
#include "../primitives.h"
#include "../utils.h"
#include "../algorithms/reduction.h"

namespace cukd {

namespace device {
__global__
void
create_chunks_kernel(device::NodeChunkArray arr, int* offsets, int n_nodes,
                     int max_chunk_size);
__global__
void
tag_triangles_left_right_kernel(device::NodeChunkArray active, int n_elements,
                                int * tag);
__global__
void
element_clipping_kernel(int n_parent_nodes, device::NodeChunkArray active,
                        int* p_split_axis, float* p_split_pos,
                        DevTriangleArray tris, int n_left);
__global__
void
update_parent_aabbs_kernel(int n_nodes, device::NodeChunkArray active,
                           device::NodeChunkArray next);
__global__
void
tag_triangles_by_node_tag_kernel(device::NodeChunkArray nca, int* node_tags,
                                 int* tags);
__global__
void
element_aabb_boundary_planes_kernel(device::NodeChunkArray nca, int nca_n_nodes,
                                    float* boundaries, int* dirs);

__global__
void
determine_empty_space_cut_kernel(int dir, int n_nodes,
                                 DevAABBArray parent_aabb,
                                 DevAABBArray node_aabb,
                                 int* cut_dir);
}  // namespace device

/**********************************************************************************
 *
 * NodeChunkArray implementation
 *
 **********************************************************************************/

typedef thrust::tuple<int, int, float, AABBArray::AABBTuple> NodeTuple;
typedef thrust::tuple<thrust::device_vector<int>::iterator,
        thrust::device_vector<int>::iterator,
        thrust::device_vector<float>::iterator,
        thrust::device_vector<int>::iterator,
        AABBArray::AABBIterator, AABBArray::AABBIterator> NodeIteratorTuple;

NodeChunkArray::NodeChunkArray() : n_chunks_(0) {}

void
NodeChunkArray::init_root_node(int n_elements, AABBArray & tri_aabbs,
                               const UAABB & root_aabb) {
    resize_nodes(1);
    resize_elements(n_elements);
    tri_aabbs.copy(triangle_aabbs);
    thrust::sequence(element_idx.begin(), element_idx.end());
    node_size.set(0, n_elements_);
    node_element_first_idx.set(0,0);
    parent_aabb.minima.set(0, root_aabb.minimum);
    parent_aabb.maxima.set(0, root_aabb.maximum);
    depth.set(0,0);
}

void
NodeChunkArray::empty_space_tags(DevVector<int> & cut_dirs,
                                 DevVector<int> & n_empty_nodes) {
    // Determine if empty space cuts for the current nodes in
    // the active lsit are possible.
    cut_dirs.resize(n_nodes_ + 1, 0);
    n_empty_nodes.resize(n_nodes_ + 1);

    DevAABBArray dev_parent_aabb = parent_aabb.dev_array();
    DevAABBArray dev_node_aabb = node_aabb.dev_array();

    for(int axis = 0; axis < 3; ++axis)
        determine_empty_space(n_nodes_, axis,
                              dev_parent_aabb, dev_node_aabb,
                              cut_dirs.pointer());

    // Compute the number of empty split nodes for each node.
    // n_empty_nodes[n_nodes + 1] is the total number of empty split
    // nodes at this pass.
    CountBitsFunctor countbits;
    thrust::exclusive_scan(
            thrust::make_transform_iterator(cut_dirs.begin(), countbits),
            thrust::make_transform_iterator(cut_dirs.end(), countbits),
            n_empty_nodes.begin());
};

void
NodeChunkArray::divide_in_chunks() {
    // compute the number of chunks for each node, such that each
    // chunk holds at most MAX_ELEMENTS_PER_CHUNK elements
    n_chunks_per_node.clear();
    n_chunks_per_node.resize(n_nodes_);

    thrust::transform(node_size.begin(),
                      node_size.begin() + n_nodes_,
                      n_chunks_per_node.begin(),
                      IntegerDivide(MAX_ELEMENTS_PER_CHUNK));
    int n_new_chunks = thrust::reduce(n_chunks_per_node.begin(),
                                      n_chunks_per_node.end());
    resize_chunks(n_new_chunks);

    // compute the indices to the element list per chunk
    first_chunk_idx.clear();
    first_chunk_idx.resize(n_nodes_);
    thrust::exclusive_scan(n_chunks_per_node.begin(),
                           n_chunks_per_node.end(),
                           first_chunk_idx.begin());

    dim3 blocks = dim3(256, 1, 1);
    dim3 grid = dim3(IntegerDivide(blocks.x)(n_nodes_),1,1);
    device::create_chunks_kernel<<<grid, blocks>>>(dev_array(),
                                                  first_chunk_idx.pointer(),
                                                   n_nodes_, MAX_ELEMENTS_PER_CHUNK);
    CUT_CHECK_ERROR("create_chunks_kernel failed");
}

void
NodeChunkArray::chunk_node_reduce_aabbs() {
    int* first_idx_ptr = chunk_element_first_idx.pointer();
    int* size_ptr = chunk_size.pointer();

    DevAABBArray tri_aabb = triangle_aabbs.dev_array();
    DevAABBArray c_aabb = chunk_aabb.dev_array();

    chunk_reduce<256, UFloat4, MinReductionMethod<UFloat4> >(
            tri_aabb.minima, c_aabb.minima,
            n_chunks_, first_idx_ptr, size_ptr);
    chunk_reduce<256, UFloat4, MaxReductionMethod<UFloat4> >(
            tri_aabb.maxima, c_aabb.maxima,
            n_chunks_, first_idx_ptr, size_ptr);

    out_keys.clear();
    out_keys.resize(n_nodes());

    thrust::equal_to<int> equal_keys;

    thrust::reduce_by_key(node_idx.begin(), node_idx.begin() + n_chunks_,
                          chunk_aabb.begin(), out_keys.begin(),
                          node_aabb.begin(),
                          equal_keys, AABBArray::MinMax());
}

int
NodeChunkArray::append_by_tag(NodeChunkArray & nca, int new_nodes,
                              DevVector<int> & node_tags,
                              DevVector<int> & element_tags) {
    element_tags.resize(nca.n_elements());

    nca.divide_in_chunks();
    nca.chunk_node_reduce_aabbs();

    dim3 grid(nca.n_chunks(), 1, 1);
    dim3 blocks(256,1,1);
    device::tag_triangles_by_node_tag_kernel<<<grid, blocks>>>(nca.dev_array(),
                                                               node_tags.pointer(),
                                                               element_tags.pointer());
    CUT_CHECK_ERROR("tag_triangles_by_node_tag_kernel failed");

    int n_old_nodes = n_nodes();
    int n_old_elements = n_elements();

    resize_nodes(n_old_nodes + new_nodes);

    // copy nodes
    NodeIteratorTuple begin =
        thrust::make_tuple(nca.node_size.begin(),
                nca.split_axis.begin(),
                nca.split_position.begin(),
                nca.depth.begin(),
                nca.parent_aabb.begin(),
                nca.node_aabb.begin());
    NodeIteratorTuple end =
        thrust::make_tuple(nca.node_size.end(),
                nca.split_axis.end(),
                nca.split_position.end(),
                nca.depth.end(),
                nca.parent_aabb.end(),
                nca.node_aabb.end());
    NodeIteratorTuple result =
        thrust::make_tuple(node_size.begin() + n_old_nodes,
                split_axis.begin() + n_old_nodes,
                split_position.begin() + n_old_nodes,
                depth.begin() + n_old_nodes,
                parent_aabb.begin() + n_old_nodes,
                node_aabb.begin() + n_old_nodes);

    IsNonZero<int> is_non_zero;
    thrust::copy_if(thrust::make_zip_iterator(begin),
            thrust::make_zip_iterator(end),
            node_tags.begin(),
            thrust::make_zip_iterator(result),
            is_non_zero);

    int new_elems = thrust::reduce(node_size.begin() + n_old_nodes, node_size.end());

    resize_elements(n_old_elements + new_elems);
    // now copy the elements
    thrust::copy_if(
            thrust::make_zip_iterator(thrust::make_tuple(nca.element_idx.begin(), nca.triangle_aabbs.begin())),
            thrust::make_zip_iterator(thrust::make_tuple(nca.element_idx.end(), nca.triangle_aabbs.end())),
            element_tags.begin(),
            thrust::make_zip_iterator(thrust::make_tuple(element_idx.begin() + n_old_elements,
                                      triangle_aabbs.begin() + n_old_elements)),
            is_non_zero);

    // nodes complete, compute first element indices
    update_node_element_first_idx();

    return new_elems;
}

void NodeChunkArray::remove_by_tag(DevVector<int> & node_tags,
                                   DevVector<int> & element_tags,
                                   int n_removed_nodes, int n_removed_elements) {
    // remove nodes
    NodeIteratorTuple begin =
        thrust::make_tuple(node_size.begin(),
                           split_axis.begin(),
                           split_position.begin(),
                           depth.begin(),
                           parent_aabb.begin(),
                           node_aabb.begin());
    NodeIteratorTuple end =
        thrust::make_tuple(node_size.end(),
                           split_axis.end(),
                           split_position.end(),
                           depth.end(),
                           parent_aabb.end(),
                           node_aabb.end());
    thrust::remove_if(thrust::make_zip_iterator(begin),
                      thrust::make_zip_iterator(end),
                      node_tags.begin(), IsNonZero<int>());
    resize_nodes(node_size.size() - n_removed_nodes);

    // remove elements
    thrust::remove_if(thrust::make_zip_iterator(
                          thrust::make_tuple(element_idx.begin(), triangle_aabbs.begin())),
                      thrust::make_zip_iterator(
                          thrust::make_tuple(element_idx.end(), triangle_aabbs.end())),
                      element_tags.begin(), IsNonZero<int>());

    resize_elements(element_idx.size() - n_removed_elements);
    update_node_element_first_idx();
}

void
NodeChunkArray::compact_elements_by_tag(DevVector<int> & tags, int start_tag_index,
                                        int end_tag_index, int start_element_index,
                                        NodeChunkArray & nca) {
    typedef thrust::tuple<int, int, AABBArray::AABBTuple> TagElementAABBTuple;
    typedef thrust::tuple<thrust::device_vector<int>::iterator,
                          thrust::device_vector<int>::iterator,
                          AABBArray::AABBIterator> IteratorTuple;

    DevVector<int> temp;
    temp.resize(tags.size());

    FirstIsValue<TagElementAABBTuple, int> first_is_value(1);

    IteratorTuple begin =
            thrust::make_tuple(tags.begin() + start_tag_index,
                               element_idx.begin(),
                               triangle_aabbs.begin());
    IteratorTuple end =
            thrust::make_tuple(tags.begin() + end_tag_index,
                               element_idx.end(),
                               triangle_aabbs.end());
    IteratorTuple result =
            thrust::make_tuple(temp.begin(),
                               nca.element_idx.begin() + start_element_index,
                               nca.triangle_aabbs.begin() + start_element_index);
    thrust::copy_if(
            thrust::make_zip_iterator(begin), thrust::make_zip_iterator(end),
            thrust::make_zip_iterator(result), first_is_value);
}

void NodeChunkArray::update_node_element_first_idx() {
    thrust::exclusive_scan(node_size.begin(), node_size.end(),
                           node_element_first_idx.begin());
}

device::NodeChunkArray
NodeChunkArray::dev_array() {
    device::NodeChunkArray dev_node_chunk_array;
    dev_node_chunk_array.na = NodeArray::dev_array();

    dev_node_chunk_array.node_idx = node_idx.pointer();
    dev_node_chunk_array.chunk_size = chunk_size.pointer();
    dev_node_chunk_array.chunk_element_first_idx = chunk_element_first_idx.pointer();

    dev_node_chunk_array.node_aabb = node_aabb.dev_array();
    dev_node_chunk_array.chunk_aabb = chunk_aabb.dev_array();
    dev_node_chunk_array.parent_aabb = parent_aabb.dev_array();
    dev_node_chunk_array.triangle_aabb = triangle_aabbs.dev_array();

    return dev_node_chunk_array;
}

void
NodeChunkArray::tag_triangles_left_right(DevVector<int> & tags) {
    dim3 grid(n_chunks(), 1, 1);
    dim3 blocks(256,1,1);
    device::tag_triangles_left_right_kernel<<<grid, blocks>>>(dev_array(),
                                                              n_elements(),
                                                              tags.pointer());
    CUT_CHECK_ERROR("tag_triangles_left_right_kernel failed");

}

void
NodeChunkArray::element_clipping(DevVector<int> & split_axis,
                                 DevVector<float> & split_pos,
                                 TriangleArray & tris, int n_left) {
    dim3 grid(n_chunks(), 1, 1);
    dim3 blocks(256,1,1);

    device::element_clipping_kernel<<<grid, blocks>>>(split_pos.size(),
                                                      dev_array(),
                                                      split_axis.pointer(),
                                                      split_pos.pointer(),
                                                      tris.dev_array(), n_left);
    CUT_CHECK_ERROR("element_clipping_kernel failed");
}

void
NodeChunkArray::element_aabb_boundary_planes(DevVector<float> & boundaries,
                                             DevVector<int> & dirs) {
    dim3 grid(IntegerDivide(256)(n_nodes()), 1, 1);
    dim3 blocks(256,1,1);
    device::element_aabb_boundary_planes_kernel<<<grid, blocks>>>(dev_array(),
                                                                  n_nodes(),
                                                                  boundaries.pointer(),
                                                                  dirs.pointer());
    CUT_CHECK_ERROR("copy_aabb_boundaries_kernel failed");
}

void
NodeChunkArray::update_parent_aabbs(cukd::NodeChunkArray & active) {
    dim3 grid(IntegerDivide(256)(active.n_nodes()),1,1);
    dim3 blocks(256,1,1);
    // FIXME: we have to pass active.n_nodes() separately, the value
    // on the device is invalid for some reason!
    device::update_parent_aabbs_kernel<<<grid,blocks>>>(active.n_nodes(),
                                                        active.dev_array(),
                                                        dev_array());
    CUT_CHECK_ERROR("update_parent_aabbs__kernel failed");
}


void
NodeChunkArray::determine_empty_space(int n_nodes, int dir,
                                      DevAABBArray & parent_aabb,
                                      DevAABBArray & node_aabb, int* cut_dir) {
    dim3 grid(IntegerDivide(256)(n_nodes),1,1);
    dim3 blocks(256,1,1);
    int shared_size = 3*sizeof(int) + sizeof(float);

    device::determine_empty_space_cut_kernel<<<grid,blocks,shared_size>>>(
                                                      dir, n_nodes,
                                                      parent_aabb, node_aabb,
                                                      cut_dir);
    CUT_CHECK_ERROR("determine_empty_space_cut_kernel failed");
}


}  // namespace cukd
