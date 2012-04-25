// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <cutil_inline.h>
#include "../kdtree.h"
#include "../algorithms/reduction.h"
#include "dev_structs.h"

namespace cukd {
namespace device {

struct Polygon {
    UFloat4 points[10];
    int length;
};

/**********************************************************************************
 *
 * KDTree Device functions
 *
 **********************************************************************************/

__device__ __host__ __inline__
void
clip_line(const UFloat4 & a, const UFloat4 & b, int side, int dir, float pos,
          Polygon & newpoints) {
    UFloat4 hit;
    hit.component[3] = 0;

    bool af = side*(a.component[dir] - pos) >= 0;
    bool bf = side*(b.component[dir] - pos) >= 0;
    int i;

    // both points on same side, return second point
    if(af && bf) {
        newpoints.points[newpoints.length] = b;
        newpoints.length += 1;
    }

    // clip the line
    if(af != bf) {
        float dist = (pos - a.component[dir])/(b.component[dir] - a.component[dir]);
#pragma unroll 3
        for(i = 0; i < 3; ++i)
            hit.component[i] = dist*(b.component[i] - a.component[i]) + a.component[i];

        hit.component[dir] = pos;

        newpoints.points[newpoints.length] = hit;
        newpoints.length += 1;
        // if second point is on the correct side, return it too
        if(bf) {
            newpoints.points[newpoints.length] = b;
            newpoints.length += 1;
        }
    }
}

// Standard polygon-bounding box clipping (Sutherland-Hodgman).
__device__ __host__ __inline__
void
clip_polygon(const Polygon & polygon, int side, int dir, float pos,
             Polygon & new_polygon) {
    int i;
    for(i = 0; i < polygon.length-1; ++i)
        clip_line(polygon.points[i], polygon.points[i+1], side, dir, pos,
                  new_polygon);
    clip_line(polygon.points[polygon.length-1], polygon.points[0], side, dir, pos,
              new_polygon);
}

__device__ __host__ __inline__
void
clip_polygon_to_aabb(Polygon & polygon, const UFloat4 & aabb_min,
                     const UFloat4 & aabb_max) {
    Polygon temp;

    if(aabb_max.component[0] != aabb_min.component[0]) {
        temp.length = 0;
        clip_polygon(polygon, -1.f, 0, aabb_max.component[0], temp);
        polygon.length = 0;
        clip_polygon(temp, 1.f, 0, aabb_min.component[0], polygon);
    }

    if(aabb_max.component[1] != aabb_min.component[1]) {
        temp.length = 0;
        clip_polygon(polygon, -1.f, 1, aabb_max.component[1], temp);
        polygon.length = 0;
        clip_polygon(temp, 1.f, 1, aabb_min.component[1], polygon);
    }

    if(aabb_max.component[2] != aabb_min.component[2]) {
        temp.length = 0;
        clip_polygon(polygon, -1.f, 2, aabb_max.component[2], temp);
        polygon.length = 0;
        clip_polygon(temp, 1.f, 2, aabb_min.component[2], polygon);
    }
}

/**********************************************************************************
 *
 * NodeChunkArray Kernels
 *
 **********************************************************************************/

__global__
void
create_chunks_kernel(device::NodeChunkArray arr, int* offsets, int n_nodes,
                     int max_chunk_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid < n_nodes) {
        int element_count = arr.na.node_size[tid];
        int first_element_idx = arr.na.node_element_first_idx[tid];
        int start_idx = offsets[tid];

        for(int i = 0, n_chunk_elements = 0, processed_elements = 0;
                processed_elements < element_count;
                ++i, processed_elements += n_chunk_elements) {

            n_chunk_elements = min(max_chunk_size,
                                   element_count - processed_elements);
            arr.chunk_element_first_idx[start_idx + i] =
                first_element_idx + processed_elements;
            arr.node_idx[start_idx + i] = tid;
            arr.chunk_size[start_idx + i] = n_chunk_elements;
        }
    }
}


__global__
void
tag_triangles_left_right_kernel(device::NodeChunkArray active,
                                int n_elements, int * tag) {
    __shared__ int split_dir;
    __shared__ float split_pos;
    __shared__ int n_tris;
    __shared__ int first_element_index;
    __shared__ int n_idx;
    __shared__ int chunk_id;

    int tid = threadIdx.x;

    UFloat4 tri_aabb_min;
    UFloat4 tri_aabb_max;
    int element_idx;

    if(tid == 0) {
        chunk_id = blockIdx.x;
        n_idx = active.node_idx[chunk_id];
        n_tris = active.chunk_size[chunk_id];
        split_dir = active.na.split_axis[n_idx];
        split_pos = active.na.split_position[n_idx];
        first_element_index = active.chunk_element_first_idx[chunk_id];
    }
    __syncthreads();

    if(tid < n_tris) {
        element_idx = tid + first_element_index;
        tri_aabb_min = active.triangle_aabb.minima[element_idx];
        tri_aabb_max = active.triangle_aabb.maxima[element_idx];

        tag[element_idx + n_elements] = (int) (tri_aabb_max.component[split_dir] > split_pos
                           || tri_aabb_min.component[split_dir] >= split_pos);
        tag[element_idx]  = (int) ((tri_aabb_min.component[split_dir] < split_pos)
                            || (tri_aabb_max.component[split_dir] <= split_pos));
    }
}

__global__
void
determine_empty_space_cut_kernel(int dir, int n_nodes,
                                 DevAABBArray parent_aabb,
                                 DevAABBArray node_aabb,
                                 int* cut_dir) {
    int tid = blockDim.x*blockIdx.x + threadIdx.x;
    if(tid >= n_nodes)
        return;
    float ratio = 0.12f;

    UFloat4 parent_aabb_min, parent_aabb_max, node_aabb_min, node_aabb_max;

    parent_aabb_min = parent_aabb.minima[tid];
    parent_aabb_max = parent_aabb.maxima[tid];
    node_aabb_min = node_aabb.minima[tid];
    node_aabb_max = node_aabb.maxima[tid];

    float parent_ratio_thr = ratio*(parent_aabb_max.component[dir]
                             - parent_aabb_min.component[dir]);

    if((parent_aabb_max.component[dir] - node_aabb_max.component[dir])
            - parent_ratio_thr > 0)
        cut_dir[tid] |= (1 << (2*dir));
    if((node_aabb_min.component[dir] - parent_aabb_min.component[dir])
            - parent_ratio_thr > 0)
        cut_dir[tid] |= (1 << (2*dir + 1));
}

__global__
void
element_clipping_kernel(int n_parent_nodes, device::NodeChunkArray active,
                        int* p_split_axis, float* p_split_pos,
                        DevTriangleArray tris, int n_left) {
    int chunk_id = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int split_dir;
    __shared__ float split_pos;
    __shared__ int n_tris;
    __shared__ int first_element_index;
    __shared__ int n_idx;

    bool leftright = false;
    Polygon triangle;
    triangle.length = 3;
    int element_tri_idx;
    int element_idx;
    UFloat4 tri_aabb_min;
    UFloat4 tri_aabb_max;
    int i;

    if(tid == 0) {
        n_idx = active.node_idx[chunk_id];
        n_tris = active.chunk_size[chunk_id];
        // FIXME: get rid of modulo!
        split_dir = p_split_axis[n_idx % n_parent_nodes];
        split_pos = p_split_pos[n_idx % n_parent_nodes];
        first_element_index = active.chunk_element_first_idx[chunk_id];
    }
    __syncthreads();

    if(tid < n_tris) {
        element_idx = tid + first_element_index;
        element_tri_idx = active.na.element_idx[element_idx];
        tri_aabb_min = active.triangle_aabb.minima[element_idx];
        tri_aabb_max = active.triangle_aabb.maxima[element_idx];

        leftright  = tri_aabb_min.component[split_dir] < split_pos &&
                     tri_aabb_max.component[split_dir] > split_pos;

        if(leftright) {

            for(i = 0; i < 3; ++i)
                triangle.points[i] = tris.v[i][element_tri_idx];

            if(element_idx < n_left) {
                tri_aabb_max.component[split_dir] = split_pos;
            } else {
                tri_aabb_min.component[split_dir] = split_pos;
            }

            // cut tri_aabb
            clip_polygon_to_aabb(triangle, tri_aabb_min, tri_aabb_max);

            // explicit unrolling
            tri_aabb_min = triangle.points[0];
            tri_aabb_max = triangle.points[0];

            for(i = 1; i < triangle.length; ++i) {
                tri_aabb_min.component[0] =
                    fminf(tri_aabb_min.component[0],
                          triangle.points[i].component[0]);
                tri_aabb_max.component[0] =
                    fmaxf(tri_aabb_max.component[0],
                          triangle.points[i].component[0]);
                tri_aabb_min.component[1] =
                    fminf(tri_aabb_min.component[1],
                          triangle.points[i].component[1]);
                tri_aabb_max.component[1] =
                    fmaxf(tri_aabb_max.component[1],
                          triangle.points[i].component[1]);
                tri_aabb_min.component[2] =
                    fminf(tri_aabb_min.component[2],
                          triangle.points[i].component[2]);
                tri_aabb_max.component[2] =
                    fmaxf(tri_aabb_max.component[2],
                          triangle.points[i].component[2]);
            }
            if(element_idx < n_left) {
                tri_aabb_max.component[split_dir] = split_pos;
            } else {
                tri_aabb_min.component[split_dir] = split_pos;
            }
            active.triangle_aabb.minima[element_idx] = tri_aabb_min;
            active.triangle_aabb.maxima[element_idx] = tri_aabb_max;

        }
    }
}

__global__
void
update_parent_aabbs_kernel(int n_nodes, device::NodeChunkArray active,
                            device::NodeChunkArray next) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    UFloat4 min_left, min_right, max_left, max_right;

    if(tid < n_nodes) {
        min_left = active.node_aabb.minima[tid];
        min_right = min_left;
        max_left = active.node_aabb.maxima[tid];
        max_right = max_left;

        max_left.component[active.na.split_axis[tid]] = active.na.split_position[tid];
        min_right.component[active.na.split_axis[tid]] = active.na.split_position[tid];

        next.parent_aabb.minima[tid] = min_left;
        next.parent_aabb.minima[tid + n_nodes] = min_right;
        next.parent_aabb.maxima[tid] = max_left;
        next.parent_aabb.maxima[tid + n_nodes] = max_right;

        next.na.depth[tid] = active.na.depth[tid] + 1;
        next.na.depth[tid + n_nodes] = active.na.depth[tid] + 1;
    }
}

__global__
void
tag_triangles_by_node_tag_kernel(device::NodeChunkArray nca, int* node_tags, int* tags) {
    int chunkidx = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ int current_node_tag;
    __shared__ int current_fst_element;
    __shared__ int n_chunk_elements;
    if(tid == 0) {
        current_node_tag = node_tags[nca.node_idx[chunkidx]];
        current_fst_element = nca.chunk_element_first_idx[chunkidx];
        n_chunk_elements = nca.chunk_size[chunkidx];
    }
    __syncthreads();

    int elemidx = current_fst_element + tid;
    if(tid < n_chunk_elements) {
        tags[elemidx] = current_node_tag;
    }
}

__global__
void
element_aabb_boundary_planes_kernel(device::NodeChunkArray nca, int nca_n_nodes, 
                                    float* boundaries, int* dirs) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    if(tid < nca_n_nodes) {
        int n_elements = nca.na.node_size[tid];
        int first_element = nca.na.node_element_first_idx[tid];
        int index = 6*nca.na.node_element_first_idx[tid];

        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 0;
            boundaries[index++] = nca.triangle_aabb.minima[i + first_element].vec.x;
        }
        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 0;
            boundaries[index++] = nca.triangle_aabb.maxima[i + first_element].vec.x;
        }
        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 1;
            boundaries[index++] = nca.triangle_aabb.minima[i + first_element].vec.y;
        }
        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 1;
            boundaries[index++] = nca.triangle_aabb.maxima[i + first_element].vec.y;
        }
        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 2;
            boundaries[index++] = nca.triangle_aabb.minima[i + first_element].vec.z;
        }
        for(int i = 0; i < n_elements; ++i) {
            dirs[index] = 2;
            boundaries[index++] = nca.triangle_aabb.maxima[i + first_element].vec.z;
        }

    }
}


}  // namespace device
}  // namespace cukd
