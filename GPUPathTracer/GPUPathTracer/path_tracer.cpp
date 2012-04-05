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

	launch_kernel(numSpheres, spheres, singlePassImage->numPixels, singlePassImage->pixels, rays, image->passCounter, renderCam);

	// TODO: Make a function for this (or a method---maybe Image can just be a class).
	for (int i = 0; i < image->numPixels; i++) {
		image->pixels[i] += singlePassImage->pixels[i];
	}
	image->passCounter++;

	deleteImage(singlePassImage);

	return image;
}

void PathTracer::setUpScene() {

	numSpheres = 8; // TODO: Move this!

}

void PathTracer::createDeviceData() {

    // Initialize data:

	Sphere* tempSpheres = new Sphere[numSpheres];

	Ray* tempRays = new Ray[image->numPixels];




	// TEMPORARY hard-coded spheres:

	Material red = makeEmptyMaterial();
	red.diffuseColor = make_float3(0.87, 0.15, 0.15);
	red.emittedColor = make_float3(0, 0, 0);
	red.specularColor = make_float3(1, 1, 1);
	red.specularRefractiveIndex = 1.62;

	Material green = makeEmptyMaterial();
	green.diffuseColor = make_float3(0.15, 0.87, 0.15);
	green.emittedColor = make_float3(0, 0, 0);
	green.specularColor = make_float3(1, 1, 1);
	green.specularRefractiveIndex = 1.62;

	Material white = makeEmptyMaterial();
	white.diffuseColor = make_float3(0.9, 0.9, 0.9);
	white.emittedColor = make_float3(0, 0, 0);
	//white.specularColor = make_float3(1, 1, 1);
	//white.specularRefractiveIndex = 1.05;

	Material gold = makeEmptyMaterial();
	gold.diffuseColor = make_float3(0, 0, 0);
	gold.emittedColor = make_float3(0, 0, 0);
	gold.specularColor = make_float3(0.869, 0.621, 0.027);
	gold.specularRefractiveIndex = 1000.0; // TODO: Make metal option or something!

	Material light = makeEmptyMaterial();
	light.diffuseColor = make_float3(0, 0, 0);
	light.emittedColor = make_float3(5.5, 4, 5.4);

	tempSpheres[0].position = make_float3(-0.9, 0, -0.9);
	tempSpheres[0].radius = 0.8;
	tempSpheres[0].material = red;

	tempSpheres[1].position = make_float3(0.8, 0, -0.4);
	tempSpheres[1].radius = 0.8;
	tempSpheres[1].material = green;

	tempSpheres[2].position = make_float3(-0.5, -0.4, 1.0);
	tempSpheres[2].radius = 0.4;
	tempSpheres[2].material = gold;

	tempSpheres[3].position = make_float3(1.3, 1.6, -2.3);
	tempSpheres[3].radius = 0.8;
	tempSpheres[3].material = light;

	tempSpheres[4].position = make_float3(-1.0, -0.7, 1.2);
	tempSpheres[4].radius = 0.1;
	tempSpheres[4].material = light;

	tempSpheres[5].position = make_float3(-0.5, -0.7, 1.7);
	tempSpheres[5].radius = 0.1;
	tempSpheres[5].material = light;

	tempSpheres[6].position = make_float3(0.3, -0.7, 1.4);
	tempSpheres[6].radius = 0.1;
	tempSpheres[6].material = light;

	tempSpheres[7].position = make_float3(0.9, -0.5, 1.3);
	tempSpheres[7].radius = 0.3;
	tempSpheres[7].material = white;





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


