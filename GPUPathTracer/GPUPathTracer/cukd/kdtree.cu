// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/scan.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/constant_iterator.h>
#include "kdtree.h"
#include "detail/dev_structs.h"
#include "algorithms/reduction.h"

namespace cukd {
namespace device {

/**********************************************************************************
 *
 * kdtree kernel wrappers
 *
 **********************************************************************************/

// Given a list of int and the corresponding AABBs that are
// contained in the current node's parent AABB, cut off the empty
// space at the node's AABB boundary. The split axes and
// split_positions as well as the indices to newly created empty nodes
// and final median splits of the node's AABBs in the Tree's
// NodeChunkArray are returned. The new indices for the final median
// splits of active list are returnes in active_indices, such that the
// tree NodeChunkArray can be updated easily.
void append_empty_nodes(cukd::NodeChunkArray & active_nca, cukd::KDTreeNodeArray & tree_nca,
                   int old_tree_nodes, DevVector<int> & cut_dirs,
                   DevVector<int> & offsets, DevVector<int> & active_indices);


// Splits small nodes at the best split plane into the NodeChunkArray
// next and the tree. If the current node is a leaf, only the tree is
// updated with the corresponding elements
void split_small_nodes(cukd::SplitCandidateArray & sca, DevVector<int> & split_indices,
                       DevVector<int> & leaf_tags, DevVector<int> & split_index_diff,
                       cukd::SmallNodeArray & active, int old_small_nodes,
                       cukd::SmallNodeArray & next, cukd::KDTreeNodeArray & tree,
                       int old_tree_nodes, int old_n_leaves);
__global__
void
preorder_bottom_up_kernel(device::KDTreeNodeArray tree, int tree_n_nodes,
                          int level, unsigned int* preorder_node_sizes);

__global__
void
preorder_top_down_kernel(device::KDTreeNodeArray tree, int tree_n_nodes,
                         int level, unsigned int* preorder_node_sizes,
                         unsigned int* addresses, unsigned int* preorder_tree);

__device__
int
ray_traverse_device(const cukdRay & ray, const UAABB & root, unsigned int *preorder,
                    DevTriangleArray & tri_vertices,
                    float & alpha, float & x1, float & x2, float & cost);
__global__
void
ray_bunch_traverse_kernel(int width, int height, DevRayArray rays, UAABB root,
                          unsigned int *preorder_tree,
                          DevTriangleArray triangles, int* hits, int* costs,
                          float* alphas, float* x1s, float* x2s);

}  // namespace device

/**********************************************************************************
 *
 * KDTree implementation
 *
 **********************************************************************************/

KDTree::KDTree(UAABB & tree_aabb, TriangleArray & tris, int small_threshold = 64)
        : root_aabb(tree_aabb), triangles(tris), _small_threshold(small_threshold), max_depth_(0) {
    active_nca.init_root_node(tris.size(), tris.aabbs, tree_aabb);
}

void
KDTree::create() {
    // large node stage
    active_nca.divide_in_chunks();
    active_nca.chunk_node_reduce_aabbs();
    // set tight bounding box as tree bounding box
    root_aabb.minimum = active_nca.node_aabb.minima.get_at(0);
    root_aabb.maximum = active_nca.node_aabb.maxima.get_at(0);

    while(active_nca.n_nodes() != 0) {
        large_nodes_process(active_nca, next_nca);
        active_nca = next_nca;
    }
    tree_nca.update_children_small();

    // small nodes determined, perform SAH splits
    SplitCandidateArray sca;
    sca.split_candidates(small_nca);

    // set initial triangle sets
    small_nca.complete();

    int tree_leaves = 0;
    SmallNodeArray next_small_nca;
    while(small_nca.n_nodes() != 0) {
        tree_leaves = small_nodes_process(small_nca, next_small_nca, sca, tree_leaves);
        small_nca = next_small_nca;
    }
    max_depth_ = tree_nca.max_depth();
}

void
KDTree::large_nodes_process(NodeChunkArray & active, NodeChunkArray & next) {
    // perform empty space and median cutting and udpate the tree accordingly
    active_indices.resize(active_nca.n_nodes());

    // TODO: redundant?
    active.divide_in_chunks();
    active.chunk_node_reduce_aabbs();

    n_empty_nodes.clear();
    cut_dirs.clear();

    active.empty_space_tags(cut_dirs, n_empty_nodes);

    int new_nodes = n_empty_nodes.get_at(active.n_nodes());
    int old_tree_nodes = tree_nca.n_nodes();

    tree_nca.resize_nodes(old_tree_nodes + new_nodes + active.n_nodes());

    device::append_empty_nodes(active, tree_nca, old_tree_nodes, cut_dirs,
                               n_empty_nodes, active_indices);

    // now split the nodes and elements into next list
    int n_left_elems = split_into_children(active, next, active_indices);

    // TODO: needed?
    next.divide_in_chunks();
    next.chunk_node_reduce_aabbs();

    // next nodes are done, perform triangle aabb clipping
    next.element_clipping(active.split_axis, active.split_position, triangles,
                          n_left_elems);

    next.update_node_element_first_idx();
    next.divide_in_chunks();
    next.chunk_node_reduce_aabbs();

    remove_small_nodes(next);
}

int
KDTree::split_into_children(NodeChunkArray & active,
                            NodeChunkArray & next,
                            DevVector<int> & active_indices) {
    // first tag the triangles according to the position left and
    // right of the split plane, repsectively
    next.resize_nodes(2*active.n_nodes());
    tags.clear();
    int elems = 2*active.n_elements();
    tags.resize(elems);

    active.tag_triangles_left_right(tags);

    // count the tags and compute new node sizes and element offsets
    int* first_idx_ptr = active.chunk_element_first_idx.pointer();
    int* size_ptr = active.chunk_size.pointer();
    chunk_sums.clear();
    chunk_sums.resize(2*active.n_chunks());

    // unsegmented per chunk
    chunk_reduce<256, int, SumReductionMethod<int> >(
            tags.pointer(), chunk_sums.pointer(),
            active.n_chunks(), first_idx_ptr, size_ptr);
    chunk_reduce<256, int, SumReductionMethod<int> >(
            tags.pointer() + active.n_elements(),
            chunk_sums.pointer() + active.n_chunks(),
            active.n_chunks(), first_idx_ptr, size_ptr);

    // segmented per node
    typedef thrust::tuple<int, int> IntPair;
    typedef thrust::tuple<thrust::device_vector<int>::iterator,
                          thrust::device_vector<int>::iterator> IntPairIterator;
    IntPairIterator chunks_begin =
         thrust::make_tuple(chunk_sums.begin(),
                            chunk_sums.begin() + active.n_chunks());
    IntPairIterator nodes_begin =
         thrust::make_tuple(next.node_size.begin(),
                            next.node_size.begin() + active.n_nodes());
    thrust::equal_to<int> equal;
    AddPair<IntPair> AddIntPair;
    thrust::reduce_by_key(active.node_idx.begin(),
                          active.node_idx.begin() + active.n_chunks(),
                          thrust::make_zip_iterator(chunks_begin),
                          out_keys.begin(),
                          thrust::make_zip_iterator(nodes_begin),
                          equal,
                          AddIntPair);


    next.update_node_element_first_idx();

    // copy parent aabbs and resize next_nca
    next.update_parent_aabbs(active);

    int new_elements = next.node_element_first_idx.get_at(2*active.n_nodes() - 1)
                       + next.node_size.get_at(2*active.n_nodes() - 1);
    int elems_left = next.node_element_first_idx.get_at(active.n_nodes());
    next.resize_elements(new_elements);

    // get rid of elements tagged with zero (compactify)
    active.compact_elements_by_tag(tags, 0, active.n_elements(), 0, next);
    active.compact_elements_by_tag(tags, active.n_elements(), 2*active.n_elements(),
                                   elems_left, next);
    next.update_node_element_first_idx();

    return elems_left;
}

void
KDTree::remove_small_nodes(NodeChunkArray & nca) {
    thrust::less_equal<int> lessEqual;
    // mark nodes with <= _small_threshold elements and compute the offsets
    small_tag.clear();
    small_tag.resize(nca.n_nodes());
    thrust::transform(nca.node_size.begin(), nca.node_size.end(),
                      thrust::constant_iterator<int>(_small_threshold),
                      small_tag.begin(), lessEqual);

    child_diff.clear();
    child_diff.resize(nca.n_nodes());
    thrust::inclusive_scan(small_tag.begin(), small_tag.end(),
                           child_diff.begin());

    // move small nodes into small_nca
    int small_nodes = child_diff.get_at(nca.n_nodes() - 1);
    if(small_nodes != 0) {
        tree_nca.update_tree_children_from_small(active_nca.n_nodes(), small_nca.n_nodes(),
                                             small_tag, child_diff, active_indices);
        small_elem_tags.clear();
        int small_elements = small_nca.append_by_tag(nca, small_nodes, small_tag,
                                                     small_elem_tags);
        nca.remove_by_tag(small_tag, small_elem_tags, small_nodes, small_elements);
    }
}

struct TagSmallLeaves {
    template<typename T>
    __device__
    void operator()(T tuple) {
        thrust::get<2>(tuple) = 0;
        int count = CountBitsFunctor()(thrust::get<0>(tuple));
        if(thrust::get<1>(tuple) >= count) {
            thrust::get<2>(tuple) = 1;
            thrust::get<3>(tuple) = count;
        }
    }
};

int
KDTree::small_nodes_process(SmallNodeArray & active, SmallNodeArray & next,
                            SplitCandidateArray & sca, int init_leaf_nodes) {
    DevVector<int> split_idx;
    DevVector<float> split_cost;
    split_idx.resize(active.n_nodes());
    split_cost.resize(active.n_nodes());

    // this should not return the splitting index, but the position,
    // so we don't need uncoalesced access later on!!
    active.best_split_SAH(sca, split_idx, split_cost);

    // determine leaf count to position the nodes in the next and tree
    // ncas
    DevVector<int> marks;
    DevVector<int> n_elements;
    marks.resize(active.n_nodes());
    n_elements.resize(active.n_nodes());
    typedef thrust::tuple<UInt64, float, int> tup;
    typedef thrust::tuple<thrust::device_vector<UInt64>::iterator,
                          thrust::device_vector<float>::iterator,
                          thrust::device_vector<int>::iterator,
                          thrust::device_vector<int>::iterator> tupit;

    tupit begin = thrust::make_tuple(active.element_bits.begin(),
                                     split_cost.begin(),
                                     marks.begin(),
                                     n_elements.begin());
    tupit end   = thrust::make_tuple(active.element_bits.end(),
                                     split_cost.end(),
                                     marks.end(),
                                     n_elements.end());
    thrust::for_each(thrust::make_zip_iterator(begin),
                      thrust::make_zip_iterator(end),
                      TagSmallLeaves());

    // compute the offsets to left and right children
    DevVector<int> mark_offsets;
    mark_offsets.resize(marks.size());
    thrust::inclusive_scan(marks.begin(), marks.end(), mark_offsets.begin());

    int old_tree_n_nodes = tree_nca.n_nodes();
    // update leaves in tree
    std::pair<int, int> leaves =
        tree_nca.update_leaves(small_nca, sca, n_elements, marks,
                mark_offsets);

    // Split nodes and add them to the next list. Store information of
    // splits in tree
    int n_leaf_nodes = leaves.second;
    int n_next_nodes = 2*(active.n_nodes() - n_leaf_nodes);
    next.clear();
    next.resize_nodes(n_next_nodes);
    next.resize_element_bits(n_next_nodes);

    device::split_small_nodes(sca, split_idx, marks, mark_offsets, active,
                              small_nca.n_nodes(), next, tree_nca, old_tree_n_nodes,
                              init_leaf_nodes);

    return n_leaf_nodes + init_leaf_nodes;
}


// almost good, problems with empty space cuts
void KDTree::preorder() {
    DevVector<unsigned int> preorder_node_sizes;
    preorder_node_sizes.resize(tree_nca.n_nodes(), 0);

    dim3 grid(IntegerDivide(256)(tree_nca.n_nodes()), 1, 1);
    dim3 blocks(256,1,1);
    for(int i = max_depth_; i >= 0; --i) {
        device::preorder_bottom_up_kernel<<<grid,blocks>>>(tree_nca.dev_array(),
                                                           tree_nca.n_nodes(), i,
                                                           preorder_node_sizes.pointer());
    }

    DevVector<unsigned int> addresses;
    addresses.resize(tree_nca.n_nodes(), 0);
    preorder_tree.resize(preorder_node_sizes.get_at(0), 0);

    for(int i = 0; i <= max_depth_; ++i) {
        device::preorder_top_down_kernel<<<grid,blocks>>>(tree_nca.dev_array(),
                                                          tree_nca.n_nodes(), i,
                                                          preorder_node_sizes.pointer(),
                                                          addresses.pointer(),
                                                          preorder_tree.pointer());
    }

};

void
KDTree::print() {
    tree_nca.print();
}

void
KDTree::print_preorder() {
    std::vector<unsigned int> poth;
    thrust::copy(preorder_tree.begin(), preorder_tree.end(), std::back_inserter(poth));

    std::cout << "*** Preorder of k-d tree ***" << std::endl;
    int i = 0;
    int elem_count = 0;
    while(i < poth.size()) {
        std::cout << "node index: " << i << std::endl;
        bool is_leaf = (leaf_mask & poth[i]) != 0;
        if(is_leaf) {
            unsigned int n_elements = n_element_mask & poth[i];
            std::cout << "  leaf node, elements: " << n_elements << std::endl;
            for(int j = 0; j < n_elements; ++j) {
                i++;
                std::cout << "    elem: " << poth[i] << std::endl;
            }
        } else {
            unsigned int right_node = right_node_mask & poth[i];
            bool left_empty = (left_empty_mask & poth[i]) != 0;
            bool right_empty = (right_empty_mask & poth[i]) != 0;
            std::cout << "  empty nodes: " << left_empty << "/" << right_empty << std::endl;
            std::cout << "  right node: " << right_node;
            if(right_empty)
                std::cout << " (ignore)";
            std::cout << std::endl;
            int split_axis = (split_axis_mask & poth[i]) >> split_axis_shift;
            i++;
            float split_position = *(float*) & poth[i];
            std::cout << "  split_plane: " << split_axis << " @ " <<
                split_position<< std::endl;
        }
        i++;
    };
    std::cout << "elements: " << elem_count << std::endl;
};

KDTreeHost
KDTree::to_host() {
    KDTreeHost result;
    thrust::copy(tree_nca.split_axis.begin(), tree_nca.split_axis.end(),
                 std::back_inserter(result.split_axis));
    thrust::copy(tree_nca.split_position.begin(), tree_nca.split_position.end(),
                 std::back_inserter(result.split_position));
    thrust::copy(tree_nca.left_nodes.begin(), tree_nca.left_nodes.end(),
                 std::back_inserter(result.left_nodes));
    thrust::copy(tree_nca.right_nodes.begin(), tree_nca.right_nodes.end(),
                 std::back_inserter(result.right_nodes));

    std::vector<UFloat4> min, max;
    thrust::copy(small_nca.node_aabb.minima.begin(), small_nca.node_aabb.minima.end(),
                 std::back_inserter(min));
    thrust::copy(small_nca.node_aabb.maxima.begin(), small_nca.node_aabb.maxima.end(),
                 std::back_inserter(max));

    for(int i = 0; i < min.size(); ++i) {
        UAABB aabb;
        aabb.minimum = min[i];
        aabb.maximum = max[i];
        result.small_node_aabbs.push_back(aabb);
    }
    int total_elements = tree_nca.n_elements();
    thrust::copy(tree_nca.element_idx.begin(),tree_nca.element_idx.begin() + total_elements,
                 std::back_inserter(result.element_idx));
    thrust::copy(tree_nca.node_element_first_idx.begin(),
            tree_nca.node_element_first_idx.end(),
            std::back_inserter(result.node_element_first_idx));
    int cells = tree_nca.n_leaves();
    thrust::copy(tree_nca.node_size.begin(),
            tree_nca.node_size.begin() + cells,
            std::back_inserter(result.element_size));

    thrust::copy(tree_nca.leaf_idx.begin(),
            tree_nca.leaf_idx.end(),
            std::back_inserter(result.leaf_index));

    return result;
}

void
KDTree::ray_bunch_traverse(int width, int height, RayArray & rays,
                           DevVector<int> & hits, DevVector<int> & costs,
                           DevVector<float> & alpha, DevVector<float> & x1, 
                           DevVector<float> & x2) {
    dim3 grid(IntegerDivide(256)(width*height), 1, 1);
    dim3 blocks(256, 1, 1);

    device::ray_bunch_traverse_kernel<<<grid, blocks>>>(
           width, height, rays.dev_array(), root_aabb,
           preorder_tree.pointer(), triangles.dev_array(),
           hits.pointer(), costs.pointer(), alpha.pointer(),
           x1.pointer(), x2.pointer());
    CUT_CHECK_ERROR("ray_bunch_traverse_kernelray_bunch_traverse_kernel failed");
};

}  // namespace cukd
