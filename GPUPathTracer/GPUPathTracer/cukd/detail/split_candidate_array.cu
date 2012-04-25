// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <thrust/iterator/constant_iterator.h>
#include <thrust/for_each.h>
#include "split_candidate_array.h"

namespace cukd {
namespace device {

__global__
void
left_right_split_candidates_kernel(device::NodeChunkArray nca,
                                   device::SplitCandidateArray sca) {
    int node = blockIdx.x;
    int tid = threadIdx.x;
    int current_element;
    int current_split;
    int axis;
    float split;
    UInt64 left_elems, right_elems;
    bool left, right;
    int i;

    __shared__ int n_elements;
    __shared__ int first_element;
    __shared__ int first_split;
    __shared__ UFloat4 tri_min[64];
    __shared__ UFloat4 tri_max[64];
    if(tid == 0) {
        n_elements = nca.na.node_size[node];
        first_element = nca.na.node_element_first_idx[node];
        first_split = sca.first_split_idx[node];
    }
    __syncthreads();

    current_element = first_element + tid;
    current_split = first_split + tid;

    // populate triangle boundaries in shared mem
    if(tid < n_elements) {
        tri_min[tid] = nca.triangle_aabb.minima[current_element];
        tri_max[tid] = nca.triangle_aabb.maxima[current_element];
    }
    __syncthreads();

    // TODO: improve if triangle aabb boundary is the same as
    // splitting candidate
    if(tid < 6*n_elements) {
        axis = (tid / (2*n_elements));
        split = sca.split_position[current_split];
        left_elems = 0;
        right_elems = 0;
        for(i = 0; i < n_elements; ++i) {
            left = tri_min[i].component[axis] < split;
            right = tri_max[i].component[axis] >= split;
            left_elems |= (UInt64) left << i;
            right_elems |= (UInt64) right << i;
        }
        __syncthreads();
        sca.left_elements[current_split] = left_elems;
        sca.right_elements[current_split] = right_elems;
    }
}

}  // namespace device

struct MultiplyDoublet {
    template<typename Tuple>
    __device__
    void operator()(const Tuple & tuple) {
        thrust::get<3>(tuple) = thrust::get<0>(tuple)*thrust::get<2>(tuple);
        thrust::get<4>(tuple) = thrust::get<1>(tuple)*thrust::get<2>(tuple);
    }
};

void
SplitCandidateArray::split_candidates(NodeChunkArray & nca) {
    // copy element AABB boundaries ordered by axis
    resize(nca.n_nodes(), nca.n_elements());

    thrust::for_each(thrust::make_zip_iterator(
                         thrust::make_tuple(
                             nca.node_element_first_idx.begin(),
                             nca.node_size.begin(),
                             thrust::constant_iterator<int>(6),
                             first_split_idx.begin(),
                             split_sizes.begin())),
                     thrust::make_zip_iterator(
                         thrust::make_tuple(
                             nca.node_element_first_idx.end(),
                             nca.node_size.end(),
                             thrust::constant_iterator<int>(6),
                             first_split_idx.end(),
                             split_sizes.end())),
                     MultiplyDoublet());

    thrust::copy(nca.element_idx.begin(),
                 nca.element_idx.end(),
                 element_idx.begin());
    thrust::copy(nca.node_element_first_idx.begin(),
                 nca.node_element_first_idx.end(),
                 first_element_idx.begin());

    nca.element_aabb_boundary_planes(split_position, split_direction);

    dim3 grid(nca.n_nodes(), 1, 1);
    dim3 blocks(6*64,1,1);
    device::left_right_split_candidates_kernel<<<grid, blocks>>>(nca.dev_array(),
                                                                 dev_array());
    CUT_CHECK_ERROR("left_right_split_candidates_kernel failed");
}

}  // namespace cukd
