#ifndef CUDA_SAFE_CALL_H
#define CUDA_SAFE_CALL_H



#define CUDA_SAFE_CALL( call) {										 \
cudaError err = call;                                                    \
if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
            __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
} }



#endif // CUDA_SAFE_CALL_H