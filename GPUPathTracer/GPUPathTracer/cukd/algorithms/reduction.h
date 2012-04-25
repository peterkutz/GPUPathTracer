// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef CUTL_REDUCTION_H
#define CUTL_REDUCTION_H

#include <vector_types.h>
#include <vector_functions.h>
#include <cutil_math.h>
#include <cutil_inline.h>
#include <iostream>
#include <float.h>
#include "shared_mem.h"
#include "../utils.h"

/**********************************************************************************
 *
 * Reductions
 *
 **********************************************************************************/

// Reduction of chunks stored in a single continuous array
//
// element_list is an array organized in chunks of a fixed size
// chunk_size. first_index_list indexes by chunk the first element in
// element_list and chunk_lengths gives the number of elements in the
// current chunk. chunks may contain <= chunk_size elements. All
// indices > chunk_lengths will be set to the neutral_element of Method.
//
// chunk_size must be power of two 
template<unsigned int chunk_size, typename T, class Method>
void chunk_reduce(T* element_list, T* output, int n_chunks,
                  int* first_index_list, int* chunk_lengths);

// Very naive segmented reduction, requires a multiple of 256
// elements for now!!!
template<typename T, class Method>
void segmented_reduce(T* values, int n_values, int* keys, int n_keys, T* result);

// Standard unsegmented reduction in a single block
//
// Custom implementation mainly for reductions of chunks stored in a
// continuous array.
// (Based on Mark Harris' "Optimizing Parallel Reduction in CUDA",
// Lifted assumption of fixed warp size)
template <unsigned int block_size, typename T, class Method>
__device__
T reduction_device(T* sarr);

// Reduce a set of partitioned data of n_elements lenght < size within
// a single block. Moreover, when using Min/MaxReductionMethod<> index
// returns the index to the min/max element in the input array.
template<unsigned int size, unsigned int tuple_length, typename T, class Method>
__inline__ __device__
T partition_reduction_device(T* input, int* offset, int n_elements, int* index);

/**********************************************************************************
 *
 * Methods and internals
 *
 **********************************************************************************/

// cuda doesn't allow abstract base classes, loose definition of
// operators. Specialization allows different operators, strides, etc.

template<typename T>
class SumReductionMethod {
    public:
        inline static __host__ __device__
        T reduction_operator(const  T & x, const  T & y) {
            return x + y;
        }
        inline static __host__ __device__
        T neutral_element() {
            return (T) 0;
        }

        inline static __host__ __device__
        int stride() {
            return 1;
        }
};

template<typename T>
class MaxReductionMethod {
    public:
        inline static __host__ __device__
        T reduction_operator(const  T & x, const  T & y) {
            return max(x,y);
        }

        inline static __host__ __device__
        T neutral_element() {
            return (T) 0;
        }

        inline static __host__ __device__
        int stride() {
            return 1;
        }
};

template<typename T>
class MinReductionMethod {
    public:
        inline static __host__ __device__
        T reduction_operator(const  T & x, const  T & y) {
            return min(x,y);
        }

        inline static __host__ __device__
        T neutral_element() {
            return (T) 0;
        }

        inline static __host__ __device__
        int stride() {
            return 1;
        }
};

#include "reduction-inl.h"

#endif  // CUTL_REDUCTION_H
