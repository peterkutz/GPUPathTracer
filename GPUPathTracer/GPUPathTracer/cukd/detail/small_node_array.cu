// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include "small_node_array.h"

namespace cukd {
namespace device {
__global__
void
compute_SAH_kernel(device::SmallNodeArray active, device::SplitCandidateArray sca,
                   int* min_sah_split_idx, float* min_sah_cost);

} // namespace device


void
SmallNodeArray::best_split_SAH(SplitCandidateArray & sca,
                               DevVector<int> & min_sah_split_pos,
                               DevVector<float> & min_sah_cost) {
    dim3 grid(n_nodes(), 1,1);
    dim3 blocks(6*64,1,1);
    device::compute_SAH_kernel<<<grid, blocks>>>(dev_array(), sca.dev_array(),
                                                 min_sah_split_pos.pointer(),
                                                 min_sah_cost.pointer());
    CUT_CHECK_ERROR("compute_SAH_kernel failed");
}

} // cukd
