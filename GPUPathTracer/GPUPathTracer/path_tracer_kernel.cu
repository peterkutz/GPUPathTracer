#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand_kernel.h"

#include "cutil_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <ctime>

#include "image.h"
#include "sphere.h"
#include "ray.h"

#include "cuda_safe_call.h"


#define BLOCK_SIZE 256 // Number of threads in a block.


__global__ void set_up_random_number_generator_kernel(curandState* states, unsigned long seed, int numPixels) {
    
	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

    curand_init(seed, pixelIndex, 0, &states[pixelIndex]);
} 


__global__ void trace_ray_kernel(int numSpheres, Sphere* spheres, int numPixels, Ray* rays, float3* notAbsorbedColors, float3* accumulatedColors, curandState* globalCurandStates) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		// Generate a random number:
		// TODO: Generate more random numbers at a time to speed this up significantly!
		curandState localCurandState = globalCurandStates[pixelIndex];
		float randomFloat = curand_uniform(&localCurandState);
		globalCurandStates[pixelIndex] = localCurandState;

		accumulatedColors[pixelIndex] = make_float3((float)pixelIndex / (float)numPixels, (float)pixelIndex / (float)numPixels / 2.0, (float)pixelIndex / (float)numPixels / 4.0);
		accumulatedColors[pixelIndex] = make_float3(randomFloat, randomFloat, randomFloat);
	}

}

extern "C"
void launch_kernel(int numSpheres, Sphere* spheres, Image* image, Ray* rays) {
	
	// Configure grid and block sizes:
	int threadsPerBlock = BLOCK_SIZE;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int blocksPerGrid = (image->numPixels + threadsPerBlock - 1) / threadsPerBlock;

	// Set up random number generator:
	// TODO: Only do this once, not every frame!
	curandState* deviceCurandStates;
    CUDA_SAFE_CALL( cudaMalloc((void**)&deviceCurandStates, image->numPixels * sizeof(curandState)) );
    set_up_random_number_generator_kernel<<<blocksPerGrid, threadsPerBlock>>>(deviceCurandStates, time(NULL), image->numPixels);

	Color* tempNotAbsorbedColors = (Color*)malloc(image->numPixels * sizeof(Color));
	Color* tempAccumulatedColors = (Color*)malloc(image->numPixels * sizeof(Color));
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void**)&notAbsorbedColors, image->numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&accumulatedColors, image->numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMemcpy( notAbsorbedColors, tempNotAbsorbedColors, image->numPixels * sizeof(Color), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( accumulatedColors, tempAccumulatedColors, image->numPixels * sizeof(Color), cudaMemcpyHostToDevice) );
	free(tempNotAbsorbedColors);
	free(tempAccumulatedColors);

	trace_ray_kernel<<<blocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, image->numPixels, rays, notAbsorbedColors, accumulatedColors, deviceCurandStates);

	// Copy the accumulated colors from the device into the host image:
	CUDA_SAFE_CALL( cudaMemcpy( image->pixels, accumulatedColors, image->numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );
	CUDA_SAFE_CALL( cudaFree( deviceCurandStates ) );

}
