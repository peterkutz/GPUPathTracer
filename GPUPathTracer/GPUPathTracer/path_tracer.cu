#include "cutil_math.h"
#include <math.h>
#include <assert.h>
#include <stdio.h>

#define BLOCK_SIZE 256 // Number of threads in a block.
#define TARGET_SPEED 50.0
#define MAX_SPEED 120.0
#define INTERACTION_RADIUS 8.131
#define SEPARATION_FORCE_RADIUS 1.112
#define SEPARATION_FORCE_MULTIPLIER 25.0
#define COHESION_FORCE_MULTIPLIER 6.0
#define HOME_DISTANCE 25.0


__global__ void trace_ray_kernel(int numSpheres, float3* spherePositions, float* sphereRadii, float3* sphereDiffuseColors, float3* sphereEmittedColors, int numPixels, float3* rayOrigins, float3* rayDirections, float3* notAbsorbedColors, float3* accumulatedColors) {

	__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	int bx = blockIdx.x;
	int tx = threadIdx.x;

	int somethingIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (somethingIndex < 999);

}

extern "C"
void launch_kernel(int numSpheres, float3* spherePositions, float* sphereRadii, float3* sphereDiffuseColors, float3* sphereEmittedColors, int numPixels, float3* rayOrigins, float3* rayDirections) {
	
	// Configure grid and block sizes and launch the kernel:
	int threadsPerBlock = BLOCK_SIZE;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;

	float3* notAbsorbedColors;
	float3* accumulatedColors;
	cudaMalloc(&notAbsorbedColors, numPixels * sizeof(float3));
	cudaMalloc(&accumulatedColors, numPixels * sizeof(float3));

	trace_ray_kernel<<<blocksPerGrid, threadsPerBlock>>>(numSpheres, spherePositions, sphereRadii, sphereDiffuseColors, sphereEmittedColors, numPixels, rayOrigins, rayDirections, notAbsorbedColors, accumulatedColors);

}
