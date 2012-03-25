#include "path_tracer.h"

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// System:
#include <stdio.h>
#include <cmath>
#include <ctime>

// CUDA:
#include <cuda_runtime.h>
#include "cutil_math.h"

#  define CUDA_SAFE_CALL( call) {                                    \
cudaError err = call;                                                    \
if( cudaSuccess != err) {                                                \
    fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
            __FILE__, __LINE__, cudaGetErrorString( err) );              \
    exit(EXIT_FAILURE);                                                  \
} }




// Necessary forward declaration:
extern "C"
void launch_kernel(int numSpheres, float3* spherePositions, float3* sphereRadii, float3* sphereDiffuseColors, float3* sphereEmittedColors, int numPixels, float3* rayOrigins, float3* rayDirections);




PathTracer::PathTracer() {

	setUpScene();
	createDeviceData();

}


PathTracer::~PathTracer() {

	deleteDeviceData();

}


void PathTracer::setImageSize(int _imageWidth, int _imageHeight) {

	imageWidth = _imageWidth;
	imageHeight = _imageHeight;

	numPixels = imageWidth * imageHeight;

	currentImage = new float3[numPixels];

}

float3* PathTracer::render() {
	runCuda();
	return currentImage;
}

void PathTracer::setUpScene() {

	numSpheres = 1;

}

void PathTracer::runCuda() {
	launch_kernel(numSpheres, spherePositions, sphereRadii, sphereDiffuseColors, sphereEmittedColors, numPixels, rayOrigins, rayDirections);
}

void PathTracer::createDeviceData() {

    // Initialize data:
	
	float3* tempSpherePositions = (float3*)malloc(numSpheres * sizeof(float3));
	float* tempSphereRadii = (float*)malloc(numSpheres * sizeof(float));
	float3* tempSphereDiffuseColors = (float3*)malloc(numSpheres * sizeof(float3));
	float3* tempSphereEmittedColors = (float3*)malloc(numSpheres * sizeof(float3));

	float3* tempRayOrigins = (float3*)malloc(numPixels * sizeof(float3));
	float3* tempRayDirections = (float3*)malloc(numPixels * sizeof(float3));

	for (int i = 0; i < numSpheres; i++) {
		tempSpherePositions[i] = make_float3(0, 0, 0);
		tempSphereRadii[i] = 3;
		tempSphereDiffuseColors[i] = make_float3(0.9, 0.8, 0.1);
		tempSphereEmittedColors[i] = make_float3(0.5, 0.5, 0.5);
	}

    for (int i = 0; i < imageHeight; ++i) {
		for (int j = 0; j < imageWidth; ++j) {
			tempRayOrigins[i * imageWidth + j] = make_float3(0, 0, -20);
			tempRayDirections[i * imageWidth + j] = make_float3(0, 0, -20);
		}
    }

    // Copy to GPU:
	CUDA_SAFE_CALL( cudaMalloc( (void**)&rayOrigins, numPixels * sizeof(float3) ) );
    CUDA_SAFE_CALL( cudaMemcpy( rayOrigins, tempRayOrigins, numPixels * sizeof(float3), cudaMemcpyHostToDevice) );

    free(tempRayOrigins);
}

void PathTracer::deleteDeviceData() {
    CUDA_SAFE_CALL( cudaFree( spherePositions ) );
    CUDA_SAFE_CALL( cudaFree( sphereRadii ) );
	CUDA_SAFE_CALL( cudaFree( sphereDiffuseColors ) );
	CUDA_SAFE_CALL( cudaFree( sphereEmittedColors ) );
	CUDA_SAFE_CALL( cudaFree( rayOrigins ) );
	CUDA_SAFE_CALL( cudaFree( rayDirections ) );

}

