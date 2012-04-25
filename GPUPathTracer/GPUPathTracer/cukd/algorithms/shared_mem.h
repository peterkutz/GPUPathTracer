// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

#ifndef SHARED_MEM_H
#define SHARED_MEM_H

/**********************************************************************************
 *
 * SharedMemory
 *
 * Allows templated shared memory in kernel.
 * Only a single SharedMemory object per kernel is allowed and the
 * shared memory size has to be specified in host code
 *
 **********************************************************************************/

template<class T>
struct SharedMemory {
    __device__ inline operator T*() {
        extern __shared__ int sarr___[];
        return (T*) sarr___;
    }
    __device__ inline operator const T*() const {
        extern __shared__ int sarr___[];
        return (T*) sarr___;
    }
};

#endif  // SHARED_MEM_H
