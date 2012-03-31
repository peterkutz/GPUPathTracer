#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cutil_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <ctime>

#include "basic_math.h"

#include "image.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"
#include <iostream>
#include <iomanip>

#include "cuda_safe_call.h"

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

#define BLOCK_SIZE 256 // Number of threads in a block.

__host__ __device__
unsigned int hash(unsigned int a)
{
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//E is eye, C is view, U is up
__global__ void raycast_from_camera_kernal(float3 E, float3 C, float3 U, float2 fov, float2 resolution, float3* accumulatedColors, int numPixels, Ray* rays){
	const float PI =3.1415926535897932384626422832795028841971;

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	//get x and y coordinates of pixel
	int y = int(pixelIndex/resolution.y);
	int x = pixelIndex - (y*resolution.y);
	
	if (validIndex) {
		//more compact version, in theory uses fewer registers but horrendously unreadable
		float3 PmE = (E+C) + (((2*(x/(resolution.x-1)))-1)*((cross(C,U)*float(length(C)*tan(fov.x*(PI/180))))/float(length(cross(C,U))))) + (((2*(y/(resolution.y-1)))-1)*((cross(cross(C,U), C)*float(length(C)*tan(-fov.y*(PI/180))))/float(length(cross(cross(C,U), C))))) -E;
		rays[pixelIndex].direction =  normalize(E + (float(200)*(PmE))/float(length(PmE)));

		//more legible version
		/*float CD = length(C);

		float3 A = cross(C,U);
		float3 B = cross(A,C);
		float3 M = E+C;
		float3 H = (A*float(CD*tan(fov.x*(PI/180))))/float(length(A));
		float3 V = (B*float(CD*tan(-fov.y*(PI/180))))/float(length(B));
		
		float sx = x/(resolution.x-1);
		float sy = y/(resolution.y-1);

		float3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
		float3 PmE = P-E;

		rays[pixelIndex].direction =  normalize(E + (float(200)*(PmE))/float(length(PmE)));
		rays[pixelIndex].origin = E;*/

		//accumulatedColors[pixelIndex] = rays[pixelIndex].direction;	//test code, should output green/yellow/black/red if correct
	}
}

__global__ void trace_ray_kernel(int numSpheres, Sphere* spheres, int numPixels, Ray* rays, float3* notAbsorbedColors, float3* accumulatedColors, unsigned long seed) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	thrust::default_random_engine rng(hash(seed)*hash(pixelIndex));
	thrust::uniform_real_distribution<float> u01(0,1);

	if (validIndex) {

		// Generate a random number:
		// TODO: Generate more random numbers at a time to speed this up significantly!
		float randomFloat =  u01(rng); 

		//accumulatedColors[pixelIndex] = make_float3((float)pixelIndex / (float)numPixels, (float)pixelIndex / (float)numPixels / 2.0, (float)pixelIndex / (float)numPixels / 4.0);
		//accumulatedColors[pixelIndex] = make_float3(randomFloat, randomFloat, randomFloat);
		//accumulatedColors[pixelIndex] = rays[pixelIndex].direction;

		accumulatedColors[pixelIndex] = make_float3(randomFloat, randomFloat, randomFloat) +  rays[pixelIndex].direction;
	}

}

extern "C"
void launch_kernel(int numSpheres, Sphere* spheres, Image* image, Ray* rays, int counter, Camera* rendercam) {
	
	// Configure grid and block sizes:
	int threadsPerBlock = BLOCK_SIZE;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int blocksPerGrid = (image->numPixels + threadsPerBlock - 1) / threadsPerBlock;

	// Set up random number generator:
	// TODO: Only do this once, not every frame!

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

	//only launch raycast from camera kernal if this is the first ever pass!
	if(counter==0){
		raycast_from_camera_kernal<<<blocksPerGrid, threadsPerBlock>>>(rendercam->position, rendercam->view, rendercam->up, rendercam->fov, rendercam->resolution, accumulatedColors, image->numPixels, rays);
	}

	trace_ray_kernel<<<blocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, image->numPixels, rays, notAbsorbedColors, accumulatedColors, counter);

	// Copy the accumulated colors from the device into the host image:
	CUDA_SAFE_CALL( cudaMemcpy( image->pixels, accumulatedColors, image->numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );

	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );

}
