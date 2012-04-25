// Copyright (c) 2012, Thomas Schutzmeier
// FreeBSD License
// See https://github.com/unvirtual/cukd/blob/master/LICENSE

template <unsigned int block_size, typename T, class Method>
__inline__ __device__
T reduction_device(T* sarr) {

    unsigned int thread = threadIdx.x;

    if (block_size >= 512) {
        if (thread < 256) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 256]);
        }
        __syncthreads();
    }
    if (block_size >= 256) {
        if (thread < 128) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 128]);
        }
        __syncthreads();
    }
    if (block_size >= 128) {
        if (thread < 64) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 64]);
        }
        __syncthreads();
    }

    // don't assume a specific warp-size!
    if (block_size >= 64) {
        if (thread < 32) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 32]);
        }
        __syncthreads();
    }
    if (block_size >= 32) {
        if (thread < 16) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 16]);
        }
        __syncthreads();
    }
    if (block_size >= 16) {
        if (thread < 8) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 8]);
        }
        __syncthreads();
    }
    if (block_size >= 8) {
        if (thread < 4) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 4]);
        }
        __syncthreads();
    }
    if (block_size >= 4) {
        if (thread < 2) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 2]);
        }
        __syncthreads();
    }
    if (block_size >= 2) {
        if (thread < 1) {
            sarr[thread] = Method::reduction_operator(sarr[thread], sarr[thread + 1]);
        }
        __syncthreads();
    }
    return sarr[0];
}

template<unsigned int size, unsigned int tuple_length, typename T, class Method>
__inline__ __device__
T partition_reduction_device(T* input, int* offset, int n_elements, int* index) {
    int tid = threadIdx.x;
    T result;
    __shared__ T temp[size];
    __shared__ int shared_index;
    if(tid == 0)
        shared_index = -1;

    if(tid < size) {
        temp[tid] = Method::neutral_element();
        if(tid < n_elements) {
            temp[tid] = Method::reduction_operator(input[tid + offset[0]],
                                                   input[tid + offset[1]]);
#pragma unroll
            for(int i = 2; i < tuple_length; ++i)
                temp[tid] = Method::reduction_operator(temp[tid],
                                                       input[tid + offset[i]]);
        }
        __syncthreads();

        result = reduction_device<size, T, Method>(temp);
        __syncthreads();

#pragma unroll
        for(int i = 0; i < tuple_length; ++i)
            if(result == input[tid + offset[i]])
                shared_index = tid + offset[i];
    }
    __syncthreads();

    if(tid == 0) {
        *index = shared_index;
    }
    return result;
}

/**********************************************************************************
 *
 * Specializations of reduction methods
 *
 **********************************************************************************/

template <>
inline __host__ __device__
float4
SumReductionMethod<float4>::reduction_operator(const float4 & x, const float4 & y) {
    float4 t1 = make_float4(x.x,x.y,x.z,x.w);
    float4 t2 = make_float4(y.x,y.y,y.z,y.w);
    return t1 + t2;
};

template <>
inline __host__ __device__
float4
SumReductionMethod<float4>::neutral_element() {
    return make_float4(0,0,0,0);
}

template <>
inline __host__ __device__
UFloat4
SumReductionMethod<UFloat4>::reduction_operator(const UFloat4 & x, const UFloat4 & y) {
    float4 t1 = make_float4(x.vec.x,x.vec.y,x.vec.z,x.vec.w);
    float4 t2 = make_float4(y.vec.x,y.vec.y,y.vec.z,y.vec.w);
    return make_ufloat4(t1 + t2);
};

template <>
inline __host__ __device__
UFloat4
SumReductionMethod<UFloat4>::neutral_element() {
    return make_ufloat4(0,0,0,0);
}


template <>
inline __host__ __device__
int
MaxReductionMethod<int>::neutral_element() {
    return INT_MIN;
}

template <>
inline __host__ __device__
float
MaxReductionMethod<float>::reduction_operator(const float &x, const float &y) {
    return fmaxf(x,y);
}

template <>
inline __host__ __device__
float
MaxReductionMethod<float>::neutral_element() {
    return -FLT_MAX;
}

template <>
inline __host__ __device__
float4
MaxReductionMethod<float4>::neutral_element() {
    return make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
}
template <>
inline __host__ __device__
float4
MaxReductionMethod<float4>::reduction_operator(const float4 & x, const float4 & y) {
    return make_float4(fmaxf(x.x,y.x),fmaxf(x.y,y.y),fmaxf(x.z,y.z), fmaxf(x.w,y.w));
}

template <>
inline __host__ __device__
UFloat4
MaxReductionMethod<UFloat4>::neutral_element() {
    return make_ufloat4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
}
template <>
inline __host__ __device__
UFloat4
MaxReductionMethod<UFloat4>::reduction_operator(const UFloat4 & x, const UFloat4 & y) {
    return make_ufloat4(fmaxf(x.vec.x,y.vec.x),fmaxf(x.vec.y,y.vec.y),fmaxf(x.vec.z,y.vec.z), fmaxf(x.vec.w,y.vec.w));
}

template <>
__host__ __device__ inline
int
MinReductionMethod<int>::neutral_element() {
    return INT_MAX;
}
template <>
inline __host__ __device__
float
MinReductionMethod<float>::reduction_operator(const  float &x, const  float &y) {
    return fminf(x,y);
}
template <>
inline __host__ __device__
float
MinReductionMethod<float>::neutral_element() {
    return FLT_MAX;
}

template <>
inline __host__ __device__
float4
MinReductionMethod<float4>::neutral_element() {
    return make_float4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
}
template <>
inline __host__ __device__
float4
MinReductionMethod<float4>::reduction_operator(const  float4 & x, const  float4 & y) {
    return make_float4(fminf(x.x,y.x),fminf(x.y,y.y),fminf(x.z,y.z), fminf(x.w,y.w));
}

template <>
inline __host__ __device__
UFloat4
MinReductionMethod<UFloat4>::neutral_element() {
    return make_ufloat4(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
}
template <>
inline __host__ __device__
UFloat4
MinReductionMethod<UFloat4>::reduction_operator(const  UFloat4 & x, const  UFloat4 & y) {
    return make_ufloat4(fminf(x.vec.x,y.vec.x),fminf(x.vec.y,y.vec.y),fminf(x.vec.z,y.vec.z), fminf(x.vec.w,y.vec.w));
}
