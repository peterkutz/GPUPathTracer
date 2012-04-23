#include "path_tracer.h"

#include "image.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"

#include "windows_include.h"

// System:
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <iostream>

// CUDA:
#include <cuda_runtime.h>
#include "cutil_math.h"

#include "cuda_safe_call.h"

#include "path_tracer_kernel.h"



PathTracer::PathTracer(Camera* cam) {

	renderCam = cam;

	image = newImage(renderCam->resolution.x, renderCam->resolution.y);

	setUpScene();

	createDeviceData();

}


PathTracer::~PathTracer() {

	deleteDeviceData();
	deleteImage(image);

}

void PathTracer::reset() {
	deleteImage(image); // Very important!
	image = newImage(renderCam->resolution.x, renderCam->resolution.y); 
}


Image* PathTracer::render() {
	Image* singlePassImage = newImage(image->width, image->height);

	launchKernel(numSpheres, spheres, singlePassImage->numPixels, singlePassImage->pixels, image->passCounter, renderCam);

	// TODO: Make a function for this (or a method---maybe Image can just be a class).
	for (int i = 0; i < image->numPixels; i++) {
		image->pixels[i] += singlePassImage->pixels[i];
	}
	image->passCounter++;

	deleteImage(singlePassImage);

	return image;
}

void PathTracer::setUpScene() {

	numSpheres = 9; // TODO: Move this!

}

void PathTracer::createDeviceData() {

    // Initialize data:

	Sphere* tempSpheres = new Sphere[numSpheres];




	// TEMPORARY hard-coded spheres:

	Material red = makeEmptyMaterial();
	red.diffuseColor = make_float3(0.87, 0.15, 0.15);
	red.specularColor = make_float3(1, 1, 1);
	red.specularRefractiveIndex = 1.491; // Acrylic.

	Material green = makeEmptyMaterial();
	green.diffuseColor = make_float3(0.15, 0.87, 0.15);
	green.specularColor = make_float3(1, 1, 1);
	green.specularRefractiveIndex = 1.491; // Acrylic.

	Material orange = makeEmptyMaterial();
	orange.diffuseColor = make_float3(0.93, 0.33, 0.04);
	orange.specularColor = make_float3(1, 1, 1);
	orange.specularRefractiveIndex = 1.491; // Acrylic.

	Material purple = makeEmptyMaterial();
	purple.diffuseColor = make_float3(0.5, 0.1, 0.9);
	purple.specularColor = make_float3(1, 1, 1);
	purple.specularRefractiveIndex = 1.491; // Acrylic.

	Material glass = makeEmptyMaterial();
	glass.specularColor = make_float3(1, 1, 1);
	glass.specularRefractiveIndex = 1.62; // Typical flint.
	glass.hasTransmission = true;

	Material white = makeEmptyMaterial();
	white.diffuseColor = make_float3(0.9, 0.9, 0.9);
	white.emittedColor = make_float3(0, 0, 0);

	Material gold = makeEmptyMaterial();
	gold.diffuseColor = make_float3(0, 0, 0);
	gold.emittedColor = make_float3(0, 0, 0);
	gold.specularColor = make_float3(0.869, 0.621, 0.027);
	gold.specularRefractiveIndex = 1000.0; // TODO: Make metal option or something!

	Material steel = makeEmptyMaterial();
	steel.diffuseColor = make_float3(0, 0, 0);
	steel.emittedColor = make_float3(0, 0, 0);
	steel.specularColor = make_float3(0.89, 0.89, 0.89);
	steel.specularRefractiveIndex = 1000.0; // TODO: Make metal option or something!

	Material light = makeEmptyMaterial();
	light.diffuseColor = make_float3(0, 0, 0);
	light.emittedColor = make_float3(10, 10, 9);

	tempSpheres[0].position = make_float3(-0.9, 0, -0.9);
	tempSpheres[0].radius = 0.8;
	tempSpheres[0].material = purple;

	tempSpheres[1].position = make_float3(0.8, 0, -0.4);
	tempSpheres[1].radius = 0.8;
	tempSpheres[1].material = glass;

	tempSpheres[2].position = make_float3(-0.5, -0.4, 1.0);
	tempSpheres[2].radius = 0.4;
	tempSpheres[2].material = glass;

	tempSpheres[3].position = make_float3(1.5, 1.6, -2.3);
	tempSpheres[3].radius = 0.4;
	tempSpheres[3].material = green;

	tempSpheres[4].position = make_float3(-1.0, -0.7, 1.2);
	tempSpheres[4].radius = 0.1;
	tempSpheres[4].material = steel;

	tempSpheres[5].position = make_float3(-0.5, -0.7, 1.7);
	tempSpheres[5].radius = 0.1;
	tempSpheres[5].material = steel;

	tempSpheres[6].position = make_float3(0.3, -0.7, 1.4);
	tempSpheres[6].radius = 0.1;
	tempSpheres[6].material = steel;

	tempSpheres[7].position = make_float3(-0.1, -0.7, 0.1);
	tempSpheres[7].radius = 0.1;
	tempSpheres[7].material = steel;

	tempSpheres[8].position = make_float3(0.9, -0.5, 1.3);
	tempSpheres[8].radius = 0.3;
	tempSpheres[8].material = orange;





    // Copy to GPU:
	CUDA_SAFE_CALL( cudaMalloc( (void**)&spheres, numSpheres * sizeof(Sphere) ) );
    CUDA_SAFE_CALL( cudaMemcpy( spheres, tempSpheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice) );

    delete [] tempSpheres;
}

void PathTracer::deleteDeviceData() {
    CUDA_SAFE_CALL( cudaFree( spheres ) );

}


