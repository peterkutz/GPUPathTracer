#include "cutil_math.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "image.h"
#include "sphere.h"
#include "ray.h"

#include "cuda_safe_call.h"


#define BLOCK_SIZE 256 // Number of threads in a block.


__global__ void trace_ray_kernel(int numSpheres, Sphere* spheres, int numPixels, Ray* rays, float3* notAbsorbedColors, float3* accumulatedColors) {

	__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int somethingIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (somethingIndex < 999);

}

extern "C"
void launch_kernel(int numSpheres, Sphere* spheres, Image* image, Ray* rays) {
	
	// Configure grid and block sizes and launch the kernel:
	int threadsPerBlock = BLOCK_SIZE;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int blocksPerGrid = (image->numPixels + threadsPerBlock - 1) / threadsPerBlock;

	Color* tempNotAbsorbedColors = (Color*)malloc(image->numPixels * sizeof(Color));
	Color* tempAccumulatedColors = (Color*)malloc(image->numPixels * sizeof(Color));
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	cudaMalloc((void**)&notAbsorbedColors, image->numPixels * sizeof(Color));
	cudaMalloc((void**)&accumulatedColors, image->numPixels * sizeof(Color));
	CUDA_SAFE_CALL( cudaMemcpy( notAbsorbedColors, tempNotAbsorbedColors, image->numPixels * sizeof(tempNotAbsorbedColors), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( accumulatedColors, tempAccumulatedColors, image->numPixels * sizeof(tempAccumulatedColors), cudaMemcpyHostToDevice) );
	free(tempNotAbsorbedColors);
	free(tempAccumulatedColors);

	trace_ray_kernel<<<blocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, image->numPixels, rays, notAbsorbedColors, accumulatedColors);

}
