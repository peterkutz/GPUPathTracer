// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <cutil_inline.h>
#include "../utils.h"
#include "../kdtree.h"
#include "../algorithms/reduction.h"
#include "dev_structs.h"
#include "../utils/intersection.h"

namespace cukd {
namespace device {

class TreeStack {
    public:
        __device__
        TreeStack() : ptr(0) {};

        __inline__ __device__
        void push(unsigned int nodeidx, float min, float max) {
            p_min[ptr] = min;
            p_max[ptr] = max;
            node[ptr] = nodeidx;
            ptr++;
        }

        __inline__ __device__
        bool pop(unsigned int & nodeidx, float & min, float & max) {
            if(ptr > 0) {
                ptr--;
                min = p_min[ptr];
                max = p_max[ptr];
                nodeidx = node[ptr];
                return true;
            }
            return false;
        }
        __inline__ __device__
        bool empty() { return (ptr == 0); }
    private:
        int ptr;
        unsigned int node[STACK_SIZE];
        float p_min[STACK_SIZE], p_max[STACK_SIZE];
};


// direct implementation of tree traversal algorithm taken from
// "Heuristic cukdRay shooting Algorithms", Vlastimil Havran
__device__
int
ray_traverse_device(const cukdRay & ray, const UAABB & root, unsigned int *preorder,
                    DevTriangleArray & tri_vertices,
                    float & alpha, float & x1, float & x2, int & cost) {
    TreeStack stack;
    int ret_tri = -1;
    cost = 0;
    float p_min = 0;
    float p_max = 0;

    unsigned int current_node = 0;
    bool is_leaf, is_empty, left_empty, right_empty, ray_split_relative,
         tri_intersect;
    unsigned int node_first_value, left_node, right_node, split_axis, node_first,
                 node_second, n_elements, index;
    float split_position, p_split, current_max, current_min, xalpha, xx1, xx2;

    if(!intersect_aabb(ray, root, p_min, p_max))
        return -1;

    current_min = p_min;
    current_max = p_max;
    alpha = current_max;

    do {
        cost++;
        stack.pop(current_node, current_min, current_max);

        node_first_value = preorder[current_node];
        is_leaf = (node_first_value & leaf_mask) != 0;
        is_empty = false;

        while(!is_leaf) {
            cost++;
            left_node = current_node + 2;
            right_node = node_first_value & right_node_mask;
            left_empty = (node_first_value & left_empty_mask) != 0;
            right_empty = (node_first_value & right_empty_mask) != 0;

            if(left_empty)
                left_node = 0;
            if(right_empty)
                right_node = 0;

            split_axis = (node_first_value & split_axis_mask) >> split_axis_shift;
            split_position = *(float*) & preorder[current_node + 1];

            p_split = (split_position - ray.origin.component[split_axis])
                * 1.f/ray.direction.component[split_axis];

            ray_split_relative = ray.origin.component[split_axis] <= split_position;
            if(ray_split_relative) {
                node_first = left_node;
                node_second = right_node;
            } else {
                node_first = right_node;
                node_second = left_node;
            }
            if(fabsf(ray.origin.component[split_axis] - split_position) < 1e-8f) {
                if(1.f/ray.direction.component[split_axis] > 0) {
                    current_node = node_second;
                } else {
                    current_node = node_first;
                }
            } else if (p_split > current_max || p_split < 0.f) {
                current_node = node_first;
            } else if (p_split < current_min) {
                current_node = node_second;
            } else {
                if(((node_second == left_node) && !left_empty)
                        || ((node_second == right_node) && !right_empty))
                    stack.push(node_second, p_split, current_max);

                current_node = node_first;
                current_max = p_split;
            }

            if(      ((current_node == left_node) && left_empty)
                  || ((current_node == right_node) && right_empty)) {
                is_empty = true;
                is_leaf = false;
                break;
            }

            node_first_value = preorder[current_node];
            is_leaf = (node_first_value & leaf_mask) != 0;
        }
        if(current_max >= p_max) {
            break;
        }

        if (is_empty) {
            continue;
        };

        n_elements = n_element_mask & preorder[current_node];
        for(int i = 1; i <= n_elements; ++i){
            cost++;
            index = preorder[current_node + i];
            Triangle tri;
            tri.v[0] = tri_vertices.v[0][index];
            tri.v[1] = tri_vertices.v[1][index];
            tri.v[2] = tri_vertices.v[2][index];
            tri_intersect = intersect_triangle(ray, tri, xalpha, xx1, xx2);
            if(tri_intersect && xalpha > 1e-8f && xalpha < alpha) {
                ret_tri = index;
                alpha  = xalpha;
                x1 = xx1;
                x2 = xx2;
            }

        }
        if(alpha < current_max) {
            break;
        }

    } while (!stack.empty());

    return ret_tri;
}

/**********************************************************************************
 *
 * KDTree Kernels
 *
 **********************************************************************************/

__global__
void
ray_bunch_traverse_kernel(int width, int height, DevRayArray rays, UAABB root,
                          unsigned int *preorder_tree,
                          DevTriangleArray triangles, int* hits, int* costs,
                          float *alphas, float* x1s, float* x2s) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    cukdRay ray;

    // TODO: return hit coordinates
    if(tid < width*height) {
        ray.origin = rays.origins[tid];
        ray.direction = rays.directions[tid];
        hits[tid] = ray_traverse_device(ray,  root, preorder_tree, triangles, 
                                        alphas[tid], x1s[tid], x2s[tid], costs[tid]);
    }
};

__global__
void
append_empty_nodes_kernel(int* cut_dir, int* offset, int* active_indices,
                          device::NodeChunkArray active, int act_n_nodes,
                          int n_tree_nodes,
                          device::KDTreeNodeArray tree) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid >= act_n_nodes)
        return;

    float aabb_data[6];

#pragma unroll 3
    for(int i = 0; i < 3; ++i) {
        aabb_data[2*i]   = active.node_aabb.maxima[tid].component[i];
        aabb_data[2*i+1] = active.node_aabb.minima[tid].component[i];
    }

    float xdiff = aabb_data[0] - aabb_data[1];
    float ydiff = aabb_data[2] - aabb_data[3];
    float zdiff = aabb_data[4] - aabb_data[5];

    // length = 4, 2 active -> first empty_space == 6 correct
    int current_node = (n_tree_nodes) + tid;
    int next_index = (n_tree_nodes) + act_n_nodes + offset[tid];
    int next_split_index = (n_tree_nodes) + act_n_nodes +
                           offset[act_n_nodes] + tid;

    int count = 0;
    int mask = 1;
    int right_cut = 0;
    int node_depth = active.na.depth[tid];

    if(cut_dir[tid] != 0) {
        // generate empty splits
        for(; mask <= 0x20; mask <<= 1, ++count) {
            if(cut_dir[tid] & mask) {

                tree.na.split_position[current_node] = aabb_data[count];
                tree.na.split_axis[current_node] = count / 2;


                // if non zero, right was cut
                right_cut = 0x15 & mask;

                tree.na.left_nodes[current_node] = 0;
                tree.na.right_nodes[current_node] = 0;
                if(right_cut != 0)
                    tree.na.left_nodes[current_node] = next_index;
                else
                    tree.na.right_nodes[current_node] = next_index;
                tree.na.depth[current_node] = node_depth++;
                current_node = next_index;;
                next_index++;
            }
        }
    }
    active_indices[tid] = current_node;

    if(xdiff > ydiff) {
        if(xdiff > zdiff) {
            tree.na.split_axis[current_node] = 0;
            tree.na.split_position[current_node] = 0.5f*xdiff + aabb_data[1];
        } else {
            tree.na.split_axis[current_node] = 2;
            tree.na.split_position[current_node] = 0.5f*zdiff + aabb_data[5];
        }
    } else {
        if(ydiff > zdiff) {
            tree.na.split_axis[current_node] = 1;
            tree.na.split_position[current_node] = 0.5f*ydiff + aabb_data[3];
        } else {
            tree.na.split_axis[current_node] = 2;
            tree.na.split_position[current_node] = 0.5f*zdiff + aabb_data[5];
        }
    }
    __syncthreads();
    active.na.split_axis[tid] = tree.na.split_axis[current_node];
    active.na.split_position[tid] = tree.na.split_position[current_node];

    //set indices to median cut children in final list
    tree.na.left_nodes[current_node] = next_split_index;
    tree.na.right_nodes[current_node] = next_split_index + act_n_nodes;
    active.na.depth[tid] = node_depth;
    tree.na.depth[current_node] = node_depth;
}

__global__
void
split_small_nodes_kernel(device::SplitCandidateArray sca, int* split_indices,
                         int* leaf_tags, int* split_index_diff, int old_small_nodes,
                         int old_tree_n_nodes,
                         device::SmallNodeArray active,
                         device::SmallNodeArray next,
                         device::KDTreeNodeArray tree,
                         int old_n_leaves
                         ) {

    int tid = blockIdx.x*blockDim.x + threadIdx.x;

    float split_position;
    int split_direction;
    int split_index;

    UFloat4 left_aabb_min, left_aabb_max, right_aabb_min, right_aabb_max;
    UInt64 node_elements, left_elements, right_elements;
    UInt64 root_idx;

    int tree_idx;
    int tree_left_child_idx = -100;
    int tree_right_child_idx = -200;

    int next_left_child_idx;
    int next_right_child_idx;

    int active_n_nodes;
    int new_left_n_nodes;

    if(tid < old_small_nodes) {
        active_n_nodes = old_small_nodes;
        new_left_n_nodes = active_n_nodes - split_index_diff[active_n_nodes-1];

        node_elements = active.element_bits[tid];
        root_idx = active.root_node_idx[tid];

        tree_idx = old_tree_n_nodes + tid;
        tree_left_child_idx = active_n_nodes + tree_idx - split_index_diff[tid];
        tree_right_child_idx = tree_left_child_idx + new_left_n_nodes;

        next_left_child_idx = tid - split_index_diff[tid];
        next_right_child_idx = tid - split_index_diff[tid] + new_left_n_nodes;

        split_index = split_indices[tid];
        split_position = sca.split_position[split_index];
        split_direction = sca.split_direction[split_index];

        if(leaf_tags[tid] == 1) {
            tree.na.left_nodes[tree_idx] = -1;
            tree.na.right_nodes[tree_idx] = -1;
            tree.na.split_position[tree_idx] = split_position;
            tree.na.split_axis[tree_idx] = split_direction;
            tree.na.depth[tree_idx] = active.na.na.depth[tid];
            tree.leaf_idx[tree_idx] = split_index_diff[tid] - 1 + old_n_leaves;
            if(active.na.na.depth[tid] > *tree.max_depth)
                *tree.max_depth = active.na.na.depth[tid];
        } else {
            tree.na.left_nodes[tree_idx] =  tree_left_child_idx;
            tree.na.right_nodes[tree_idx] = tree_right_child_idx;
            tree.na.split_position[tree_idx] = split_position;
            tree.na.split_axis[tree_idx] = split_direction;
            tree.na.depth[tree_idx] = active.na.na.depth[tid];

            left_elements = node_elements & sca.left_elements[split_index];
            right_elements = node_elements & sca.right_elements[split_index];

            left_aabb_min = active.na.node_aabb.minima[tid];
            left_aabb_max = active.na.node_aabb.maxima[tid];
            right_aabb_min = active.na.node_aabb.minima[tid];
            right_aabb_max = active.na.node_aabb.maxima[tid];

            left_aabb_max.component[split_direction] = split_position;
            right_aabb_min.component[split_direction] = split_position;

            next.na.node_aabb.minima[next_left_child_idx] = left_aabb_min;
            next.na.node_aabb.maxima[next_left_child_idx] = left_aabb_max;
            next.na.node_aabb.minima[next_right_child_idx] = right_aabb_min;
            next.na.node_aabb.maxima[next_right_child_idx] = right_aabb_max;

            next.element_bits[next_left_child_idx] = left_elements;
            next.element_bits[next_right_child_idx] = right_elements;
            next.root_node_idx[next_left_child_idx] = root_idx;
            next.root_node_idx[next_right_child_idx] = root_idx;

            next.na.na.depth[next_left_child_idx] = active.na.na.depth[tid] + 1;
            next.na.na.depth[next_right_child_idx] = active.na.na.depth[tid] + 1;
        }
    }
}

__global__
void
preorder_bottom_up_kernel(device::KDTreeNodeArray tree, int tree_n_nodes,
                       int level, unsigned int* preorder_node_sizes) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int left_node, right_node, result = 2;

    if(tid < tree_n_nodes && tree.na.depth[tid] == level) {
        left_node = tree.na.left_nodes[tid];
        right_node = tree.na.right_nodes[tid];
        if(left_node == -1 && right_node == -1) {
            result = tree.na.node_size[tree.leaf_idx[tid]] + 1;
        } else {
            if (left_node != 0) {
                result += preorder_node_sizes[left_node];
            }
            if (right_node != 0) {
                result += preorder_node_sizes[right_node];
            }
        }
        preorder_node_sizes[tid] = result;
    }
};

__global__
void
preorder_top_down_kernel(device::KDTreeNodeArray tree, int tree_n_nodes,
                         int level, unsigned int* preorder_node_sizes,
                         unsigned int* addresses, unsigned int* preorder_tree) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int left_node, right_node, address, right_index, left_index, leaf_idx;
    int first_elem_idx, i;

    if(tid < tree_n_nodes && tree.na.depth[tid] == level) {
        left_node = tree.na.left_nodes[tid];
        right_node = tree.na.right_nodes[tid];
        address = addresses[tid];
        preorder_tree[address] = 0;

        if(left_node == -1 && right_node == -1) {
            leaf_idx = tree.leaf_idx[tid];
            first_elem_idx = tree.na.node_element_first_idx[leaf_idx];
            for(i = 0; i < tree.na.node_size[leaf_idx]; ++i) {
                preorder_tree[address + i + 1] =
                    (unsigned int) tree.na.element_idx[first_elem_idx + i];
            }
            preorder_tree[address] |= leaf_mask;
            preorder_tree[address] |= (unsigned int) tree.na.node_size[leaf_idx];
        }

        if(left_node != right_node) {
            // just set the right address
            left_index = address + 2;
            right_index = address + preorder_node_sizes[left_node] + 2;

            if(left_node == 0)
                right_index = left_index;

            addresses[right_node] = right_index;
            addresses[left_node] = left_index;
            // if empty node, jump to the next free index

            preorder_tree[address + 1] =
                *(unsigned int*) &tree.na.split_position[tid];

            preorder_tree[address] = (unsigned int) right_index;
            preorder_tree[address] |=
                ((unsigned int) tree.na.split_axis[tid] << split_axis_shift);
        }
        if(left_node == 0) {
            preorder_tree[address] |= (unsigned int) left_empty_mask;
        }
        if(right_node == 0) {
            preorder_tree[address] |= (unsigned int) right_empty_mask;
        }
    }
};

/**********************************************************************************
 *
 * KDTree Kernel wrappers
 *
 **********************************************************************************/

void
append_empty_nodes(cukd::NodeChunkArray & active_nca, cukd::KDTreeNodeArray & tree_nca,
                   int old_tree_nodes,
                   DevVector<int> & cut_dirs, DevVector<int> & offsets,
                   DevVector<int> & active_indices) {
    dim3 grid(IntegerDivide(256)(active_nca.n_nodes()),1,1);
    dim3 blocks(256,1,1);

    append_empty_nodes_kernel<<<grid,blocks>>>(cut_dirs.pointer(), offsets.pointer(),
                                               active_indices.pointer(),
                                               active_nca.dev_array(),
                                               active_nca.n_nodes(),
                                               old_tree_nodes,
                                               tree_nca.dev_array());
    CUT_CHECK_ERROR("append_empty_nodes_kernel failed");
}



void split_small_nodes(cukd::SplitCandidateArray & sca, DevVector<int> & split_indices,
                       DevVector<int> & leaf_tags, DevVector<int> & split_index_diff,
                       cukd::SmallNodeArray & active, int old_small_nodes,
                       cukd::SmallNodeArray & next, cukd::KDTreeNodeArray & tree,
                       int old_tree_nodes, int old_n_leaves) {
    dim3 grid(IntegerDivide(256)(old_small_nodes),1,1);
    dim3 blocks(256,1,1);

    split_small_nodes_kernel<<<grid, blocks>>>(sca.dev_array(),
            split_indices.pointer(), leaf_tags.pointer(),
            split_index_diff.pointer(), old_small_nodes, old_tree_nodes,
            active.dev_array(), next.dev_array(), tree.dev_array(), old_n_leaves);
    CUT_CHECK_ERROR("split_small_nodes_kernel failed");
}

}  // namespace device
}  // namespace cukd
