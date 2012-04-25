// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUKD_NODE_CHUNK_ARRAY_H
#define CUKD_NODE_CHUNK_ARRAY_H

#include "../utils.h"
#include "../primitives.h"
#include "dev_structs.h"
#include "node_array.h"

namespace cukd {
#define MAX_ELEMENTS_PER_CHUNK 256

/**********************************************************************************
 *
 * NodeChunkArray
 *
 * Manages and creates chunk and node arrays for the KDTree. All
 * arrays are stored as DevVector<T> so owe can use thrust easily.
 *
 **********************************************************************************/

class NodeChunkArray : protected NodeArray {

    friend class KDTree;
    friend class SplitCandidateArray;
    public:
        NodeChunkArray();

        using NodeArray::n_nodes;
        using NodeArray::n_elements;
        int n_chunks() { return n_chunks_; };

        void resize_nodes(int nodes);
        void resize_elements(int elements);
        void resize_chunks(int chunks);

        void init_root_node(int n_elements, AABBArray & tri_aabbs,
                            const UAABB & root_aabb);

        // Clear all members without freeing memory, raw
        // pointers may still access old elements
        void clear();

        // Create a list of chunks for the stored nodes
        void divide_in_chunks();

        // Compute the per-chunk and per-node AABBs
        void chunk_node_reduce_aabbs();

        // Tag possible empty space cuts
        void empty_space_tags(DevVector<int> & cut_dir,
                              DevVector<int> & n_empty_nodes);

        // Determines if empty space in AABBs along a direction dir can be
        // cut. Every element in parent_aabb contains a smaller AABB
        // node_aabb. If the space to the left or right of node_aabb[i] along
        // the direction dir in parent_aabb[i] is larger than a given ratio,
        // cut_dir is set to a non-zero value. cut_dir serves as a bitfield:
        //    bits: zl zr yl yr xl xr (six rightmost bits)
        // where zl/zr is set if the empty space was found left/right along
        // the z-direction and the same for the x- and y-directions.
        void determine_empty_space(int n_nodes, int dir,
                                   DevAABBArray & parent_aabb,
                                   DevAABBArray & node_aabb, int* cut_dir);

        // Append another nca's nodes and elements from node tags.
        // Computes tags of corresponding elements and stores them in
        // out_element_tags.  Returns the number of appended elements
        int append_by_tag(NodeChunkArray & nca, int n_nodes,
                          DevVector<int> & node_tags,
                          DevVector<int> & out_element_tags);

        // Remove elements and nodes given node and element tags.
        // Requires the number of removed nodes and elements as input
        void remove_by_tag(DevVector<int> & node_tags, DevVector<int> & element_tags,
                           int n_removed_nodes, int n_removed_elements);

        // given a list of tags, compact current NodeChunkArray's
        // elements into result's element list
        void compact_elements_by_tag(DevVector<int> & tags,
                                     int start_tag_index, int end_tag_index,
                                     int start_element_index,
                                     NodeChunkArray & result);

        void update_node_element_first_idx();

        // Tags triangles in active according to their position relative to
        // the current split. The returned array tags is of size
        // 2*active.n_elements(). tags[i]/tags[i + active.n_elements()] is set
        // to one if the i-th element is in left/right relative to the cut,
        // zero otherwise.
        void tag_triangles_left_right(DevVector<int> & tags);

        // Updates active's triangle AABBs after clipping the corresponding
        // triangles with the current splitting plane. The passed
        // TriangleArray is not modified. Requires the number of left elements
        // tp be passed as parameter.
        void element_clipping(DevVector<int> & split_axis,
                              DevVector<float> & split_pos,
                              TriangleArray & tris, int n_left);

        // Gets the element's AABB boundary planes sorted by node, by axis and
        // left/right boundary
        // n1_x_min_1,n1_x_min_2,...,n1_y_min_1,n1_y_min_2,...,n1_z_min_1,n1_zmin_2,...
        // n2_x_min_1,n2_x_min_2,...,n2_y_min_1,n2_y_min_2,...,n2_z_min_1,n2_zmin_2,...
        void element_aabb_boundary_planes(DevVector<float> & boundaries,
                                          DevVector<int> & dirs);
        void update_parent_aabbs(cukd::NodeChunkArray & active);

        device::NodeChunkArray dev_array();

    protected:
        int n_chunks_;

        // Node and chunk AABBs. We only keep track of tight AABBs
        // containing elements, real node AABBs can be computed from
        // split_position and split_axis when needed.
        AABBArray node_aabb, chunk_aabb, parent_aabb;
        AABBArray triangle_aabbs;

        DevVector<int> node_idx;
        DevVector<int> chunk_size;
        DevVector<int> chunk_element_first_idx;

        // helper DevVectors
        DevVector<int> n_chunks_per_node;
        DevVector<int> first_chunk_idx;
        DevVector<int> out_keys;
};

inline
void
NodeChunkArray::resize_nodes(int n) {
    NodeArray::resize_nodes(n);
    node_aabb.resize(n);
    parent_aabb.resize(n);
};

inline
void
NodeChunkArray::resize_chunks(int n) {
    n_chunks_ = n;
    node_idx.resize(n);
    chunk_size.resize(n);
    chunk_element_first_idx.resize(n);
    chunk_aabb.resize(n);
};

inline
void
NodeChunkArray::resize_elements(int n) {
    NodeArray::resize_elements(n);
    triangle_aabbs.resize(n);
};

inline
void
NodeChunkArray::clear() {
    NodeArray::clear();

    n_chunks_ = 0;
    node_aabb.clear();
    chunk_aabb.clear();
    parent_aabb.clear();
    triangle_aabbs.clear();

    node_idx.clear();
    chunk_size.clear();
    chunk_element_first_idx.clear();

    n_chunks_per_node.clear();
    first_chunk_idx.clear();
    out_keys.clear();
}

}  // namespace cukd


#endif  // CUKD_NODE_CHUNK_ARRAY_H
