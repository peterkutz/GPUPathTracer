// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
//
// This software contains source code (or derivatives) provided by
// NVIDIA Corporation. See LICENSE for details.
//
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#include <iostream>
#include "shared_mem.h"
#include "reduction.h"

/**********************************************************************************
 *
 * Kernels
 *
 **********************************************************************************/

template <unsigned int chunk_size, typename T, class Method>
__global__
void chunk_reduction_kernel(T* element_list,
                            T* output,
                            int* first_index_list,
                            int* chunk_lengths) {

    T* sarr = SharedMemory<T>();
    unsigned int chunk_index = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int first_index, chunk_length;

    first_index = first_index_list[chunk_index];
    chunk_length = chunk_lengths[chunk_index];
    __syncthreads();
    if (tid < chunk_length) {
        sarr[tid] = element_list[tid + first_index];
    } else {
        sarr[tid] = Method::neutral_element();
    }
    __syncthreads();

    T result = reduction_device<chunk_size, T, Method>(sarr);

    if(tid == 0) {
        output[chunk_index] = result;
    }
}

template<typename T, class Method>
__global__
void segmented_reduction_kernel(T* values, int n_values, int* keys, int n_keys,
                                int* keyranges, T* result) {
    __shared__ int skeys[256];
    __shared__ int svalues[256];
    T* sres = SharedMemory<T>();
    int minkey = keyranges[2*blockIdx.x];
    int keydiff = keyranges[2*blockIdx.x + 1] - minkey;

    int thread = threadIdx.x;
    int index = blockIdx.x*blockDim.x + threadIdx.x;

    // load keys and values
    svalues[thread] = values[index];
    skeys[thread] = keys[index];

    // result with the proper length
    if(thread <= keydiff)
        sres[thread] = 0;

    __syncthreads();
    for(int i = 1; i < blockDim.x; i *= 2) {
        if(thread % (2*i) == 0) {
            int w0 = skeys[thread];
            int w1 = skeys[thread + i];
            if(w0 != w1) {
                sres[w1 - minkey] += svalues[thread + i];
            }
            else {
                svalues[thread] += svalues[thread + i];
            }
        }
        __syncthreads();
    }
    // atomicAdd is fine here, as there are only few of those ops per
    // thread
    if(thread <= keydiff)
        atomicAdd(&result[minkey+thread], sres[thread]);
    __syncthreads();
    if(thread == 0)
        atomicAdd(&result[skeys[0]],svalues[0]);
}

// Prepare keys for segmented reduction
void __global__ seg_reduce_keyrange_per_block(int* keys, int* ranges,
                                              int n, int blockdim) {
    int i = threadIdx.x;
    if(i < blockdim) {
        ranges[2*i] = keys[i*blockdim];
        ranges[2*i+1] = keys[(i+1)*blockdim - 1];
    }
}

/**********************************************************************************
 *
 * Kernel Wrappers
 *
 **********************************************************************************/

template<unsigned int chunk_size, typename T, class Method>
void
chunk_reduce(T* element_list, T* output, int n_chunks,
             int* first_index_list, int* chunk_lengths) {
    dim3 blocks = dim3(chunk_size, 1, 1);
    dim3 grid = dim3(n_chunks, 1, 1);
    int shared_size = chunk_size*sizeof(T);

    chunk_reduction_kernel<chunk_size, T, Method> <<<grid,blocks,shared_size>>>
        (element_list, output, first_index_list, chunk_lengths);
}

template<typename T, class Method>
void
segmented_reduce(T* values, int n_values, int* keys, int n_keys, T* result) {
    int n_threads = 256;
    int n_blocks = n_values/n_threads;
    int* dev_keyranges;
    cutilSafeCall(cudaMalloc((void**) &dev_keyranges, 2*n_blocks*sizeof(int)));

    dim3 dim_block_ranges(n_blocks,1,1);
    dim3 dim_grid_ranges(1,1,1);
    seg_reduce_keyrange_per_block<<<dim_grid_ranges, dim_block_ranges>>>(
            keys, dev_keyranges, n_values, n_threads);

    dim3 dim_block(n_threads, 1, 1);
    dim3 dim_grid(n_blocks, 1, 1);
    int smesize = n_threads*3*sizeof(int);
    segmented_reduction_kernel<T,Method><<<dim_grid, dim_block,smesize>>> (
            values, n_values, keys, n_keys, dev_keyranges, result);
    CUT_CHECK_ERROR("Segmented Reduction kernel call failed");
    cudaFree(dev_keyranges);
}

/**********************************************************************************
 *
 * Template specializations
 *
 **********************************************************************************/

template void
chunk_reduce<256, int, SumReductionMethod<int> >(
        int* element_list, int* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, float, SumReductionMethod<float> >(
        float* element_list, float* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, int, MinReductionMethod<int> >(
        int* element_list, int* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, int, MaxReductionMethod<int> >(
        int* element_list, int* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, float, MinReductionMethod<float> >(
        float* element_list, float* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, float, MaxReductionMethod<float> >(
        float* element_list, float* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, float4, MinReductionMethod<float4> >(
        float4* element_list, float4* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, float4, MaxReductionMethod<float4> >(
        float4* element_list, float4* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, UFloat4, MinReductionMethod<UFloat4> >(
        UFloat4* element_list, UFloat4* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
chunk_reduce<256, UFloat4, MaxReductionMethod<UFloat4> >(
        UFloat4* element_list, UFloat4* output, int n_chunks,
        int* first_index_list, int* chunk_lengths);

template void
segmented_reduce<int, SumReductionMethod<int> >(int* values, int n_values, int* keys,
                                                int n_keys, int* result);
