#include "path_tracer.h"

#include "image.h"
#include "sphere.h"
#include "ray.h"

#include "windows_include.h"

// System:
#include <stdio.h>
#include <cmath>
#include <ctime>

// CUDA:
#include <cuda_runtime.h>
#include "cutil_math.h"

#include "cuda_safe_call.h"

#include "path_tracer_kernel.h"



PathTracer::PathTracer() {

	image = newImage(512, 512); // TODO: Don't hard-code this.
	setUpScene();
	createDeviceData();

}


PathTracer::~PathTracer() {

	deleteDeviceData();
	deleteImage(image);

}

Image* PathTracer::render() {
	Image* singlePassImage = newImage(image->width, image->height);
	launch_kernel(numSpheres, spheres, singlePassImage, rays);
	memcpy(image->pixels, singlePassImage->pixels, image->numPixels * sizeof(Color)); // TEMP TEST.
	deleteImage(singlePassImage);
	return image;
}

void PathTracer::setUpScene() {

	numSpheres = 1;

}

void PathTracer::createDeviceData() {

    // Initialize data:
	
	Sphere* tempSpheres = new Sphere[numSpheres];

	Ray* tempRays = new Ray[image->numPixels];

	for (int i = 0; i < numSpheres; i++) {
		tempSpheres[i].position = make_float3(0, 0, 0);
		tempSpheres[i].radius = 2;
	}

    for (int i = 0; i < image->height; ++i) {
		for (int j = 0; j < image->width; ++j) {
			// TODO: Make a function for pixel array index.
			tempRays[i * image->width + j].origin = make_float3(0, 0, -20);
			tempRays[i * image->height + j].direction = make_float3(0, 0, -1);
		}
    }

    // Copy to GPU:
	CUDA_SAFE_CALL( cudaMalloc( (void**)&spheres, numSpheres * sizeof(Sphere) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&rays, image->numPixels * sizeof(Ray) ) );
    CUDA_SAFE_CALL( cudaMemcpy( spheres, tempSpheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice) );
    CUDA_SAFE_CALL( cudaMemcpy( rays, tempRays, image->numPixels * sizeof(Ray), cudaMemcpyHostToDevice) );

    delete [] tempSpheres;
	delete [] tempRays;
}

void PathTracer::deleteDeviceData() {
    CUDA_SAFE_CALL( cudaFree( spheres ) );
    CUDA_SAFE_CALL( cudaFree( rays ) );

}

