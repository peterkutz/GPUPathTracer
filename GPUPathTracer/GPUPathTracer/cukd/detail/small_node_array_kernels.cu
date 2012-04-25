// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <cutil_inline.h>
#include "../kdtree.h"
#include "../algorithms/reduction.h"
#include "dev_structs.h"

namespace cukd {
namespace device {

__device__ __inline__
float node_area(const UFloat4 & aabb_min, const UFloat4 & aabb_max) {
    UFloat4 aabb_diag;
    for(int i = 0; i < 3; ++i)
        aabb_diag.component[i] = aabb_max.component[i] - aabb_min.component[i];
    return 2.f*(aabb_diag.component[0]*aabb_diag.component[1] +
                aabb_diag.component[0]*aabb_diag.component[2] +
                aabb_diag.component[1]*aabb_diag.component[2]);
}

__global__
void
compute_SAH_kernel(device::SmallNodeArray active, device::SplitCandidateArray sca,
                   int* min_sah_split_idx, float* min_sah_cost) {
    int node_idx = blockIdx.x;
    int split_idx = threadIdx.x;

    int n_left_of_split, n_right_of_split, split_axis;
    UFloat4 left_aabb_max, right_aabb_min;
    float left_aabb_area, right_aabb_area;
    float split_pos, min_sah;
    __shared__ UInt64 triangle_set;
    __shared__ int root_idx, first_split, n_splits, min_split_idx;
    __shared__ float parent_inv_area;
    __shared__ UFloat4 aabb_min, aabb_max;
    __shared__ float axis_sah[6*64];
    __shared__ int split_axis_offset[3];

    if(split_idx == 0) {
        min_split_idx = -1;
        triangle_set = active.element_bits[node_idx];
        root_idx = active.root_node_idx[node_idx];
        first_split = sca.first_split_idx[root_idx];
        n_splits = sca.split_sizes[root_idx];
#pragma unroll 3
        for(int i = 0; i < 3; ++i)
            split_axis_offset[i] = i*n_splits/3;
        aabb_min = active.na.node_aabb.minima[node_idx];
        aabb_max = active.na.node_aabb.maxima[node_idx];
        parent_inv_area = 1.f/node_area(aabb_min, aabb_max);
    }
    __syncthreads();

    if(split_idx < n_splits) {
        n_left_of_split =
            CountBitsFunctor()(triangle_set & sca.left_elements[first_split + split_idx]);
        n_right_of_split =
            CountBitsFunctor()(triangle_set & sca.right_elements[first_split + split_idx]);

        if(n_left_of_split == 0 || n_right_of_split == 0) {
            axis_sah[split_idx] = MinReductionMethod<float>::neutral_element();
        } else {
            split_pos = sca.split_position[first_split + split_idx];
            split_axis = sca.split_direction[first_split + split_idx];

            // FIXME: inefficient, full aabb of left and right
            // children not needed
            left_aabb_max = aabb_max;
            right_aabb_min = aabb_min;
            left_aabb_max.component[split_axis] = split_pos;
            right_aabb_min.component[split_axis] = split_pos;

            left_aabb_area = node_area(aabb_min, left_aabb_max);
            right_aabb_area = node_area(right_aabb_min, aabb_max);

            axis_sah[split_idx] = parent_inv_area*(n_left_of_split*left_aabb_area
                                                + n_right_of_split*right_aabb_area) + 6.f;
        }
    }
    __syncthreads();

    min_sah = partition_reduction_device<128, 3, float, MinReductionMethod<float> >
                (&axis_sah[0], &split_axis_offset[0], n_splits/3, &min_split_idx);

    if(split_idx == 0) {
        min_sah_split_idx[node_idx] = min_split_idx + first_split;
        min_sah_cost[node_idx] = min_sah;
    }
}

}  // device
}  // cukd
