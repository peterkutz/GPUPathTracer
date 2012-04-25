#include "path_tracer.h"

#include "image.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"
#include "poly.h"

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

#include "objcore/objloader.h"

PathTracer::PathTracer(Camera* cam) {

	renderCamera = cam;

	image = newImage(renderCamera->resolution.x, renderCamera->resolution.y);

	numPolys = 0;

	//prepMesh();

	setUpScene();

	createDeviceData();

	
}


PathTracer::~PathTracer() {

	deleteDeviceData();
	deleteImage(image);

}

void::PathTracer::prepMesh(){
	obj* m = new obj();
	objLoader meshload("../diamond.obj", m);

	polys = new Poly[m->getFaces()->size()];

	for(int i=0; i<m->getFaces()->size(); i++){
		glm::vec4 p0 = m->getPoints()->operator[](m->getFaces()->operator[](i)[0]);
		glm::vec4 p1 = m->getPoints()->operator[](m->getFaces()->operator[](i)[1]);
		glm::vec4 p2 = m->getPoints()->operator[](m->getFaces()->operator[](i)[2]);
		glm::vec4 n = m->getNormals()->operator[](m->getFaceNormals()->operator[](i)[0]);
		polys[i].n = make_float3(n[0], n[1], n[2]);
		polys[i].p0 = make_float3(p0[0], p0[1], p0[2]);
		polys[i].p1 = make_float3(p1[0], p1[1], p1[2]);
		polys[i].p2 = make_float3(p2[0], p2[1], p2[2]);
	}

	numPolys = m->getFaces()->size();

	/*std::vector<Triangle> tris;
	for(int i=0; i<m->getFaces()->size(); i++){
		Triangle tri;
		for(int j = 0; j < 3; ++j) {
			glm::vec4 p = m->getPoints()->operator[](m->getFaces()->operator[](i)[j]);
			glm::vec4 n = m->getNormals()->operator[](m->getFaceNormals()->operator[](i)[j]);
            tri.v[j].f3.vec.x = p[0]; tri.v[j].f3.vec.y = p[1]; tri.v[j].f3.vec.z = p[2];
            tri.n[j].f3.vec.x = n[0]; tri.n[j].f3.vec.y = n[1]; tri.n[j].f3.vec.z = n[2];
        }
		tris.push_back(tri);
	}
	TriangleArray triarr = TriangleArray(tris);

	constructKDTree(triarr, m->getMin()[0], m->getMin()[1],  m->getMin()[2],  m->getMax()[0],  m->getMax()[1],  m->getMax()[2]);
	*/
}

void PathTracer::reset() {
	deleteImage(image); // Very important!
	image = newImage(renderCamera->resolution.x, renderCamera->resolution.y); 
}


Image* PathTracer::render() {
	Image* singlePassImage = newImage(image->width, image->height);

	launchKernel(numPolys, dev_polys, numSpheres, spheres, singlePassImage->numPixels, singlePassImage->pixels, image->passCounter, *renderCamera); // Dereference not ideal.

	// TODO: Make a function for this (or a method---maybe Image can just be a class).
	for (int i = 0; i < image->numPixels; i++) {
		image->pixels[i] += singlePassImage->pixels[i];
	}
	image->passCounter++;

	deleteImage(singlePassImage);

	return image;
}

void PathTracer::setUpScene() {

	numSpheres = 14; // TODO: Move this!

}

void PathTracer::createDeviceData() {

    // Initialize data:

	Sphere* tempSpheres = new Sphere[numSpheres];

	// TEMPORARY hard-coded spheres:

	Material red;
	SET_DEFAULT_MATERIAL_PROPERTIES(red);
	red.diffuseColor = make_float3(0.87, 0.15, 0.15);
	red.specularColor = make_float3(1, 1, 1);
	red.medium.refractiveIndex = 1.491; // Acrylic.

	Material green;
	SET_DEFAULT_MATERIAL_PROPERTIES(green);
	green.diffuseColor = make_float3(0.15, 0.87, 0.15);
	green.specularColor = make_float3(1, 1, 1);
	green.medium.refractiveIndex = 1.491; // Acrylic.

	Material orange;
	SET_DEFAULT_MATERIAL_PROPERTIES(orange);
	orange.diffuseColor = make_float3(0.93, 0.33, 0.04);
	orange.specularColor = make_float3(1, 1, 1);
	orange.medium.refractiveIndex = 1.491; // Acrylic.

	Material purple;
	SET_DEFAULT_MATERIAL_PROPERTIES(purple);
	purple.diffuseColor = make_float3(0.5, 0.1, 0.9);
	purple.specularColor = make_float3(1, 1, 1);
	purple.medium.refractiveIndex = 1.491; // Acrylic.

	Material glass;
	SET_DEFAULT_MATERIAL_PROPERTIES(glass);
	purple.diffuseColor = make_float3(0,0,0);
	glass.specularColor = make_float3(1, 1, 1);
	glass.medium.refractiveIndex = 2.42; // Typical flint.
	glass.hasTransmission = true;

	Material greenGlass;
	SET_DEFAULT_MATERIAL_PROPERTIES(greenGlass);
	greenGlass.specularColor = make_float3(1, 1, 1);
	greenGlass.hasTransmission = true;
	greenGlass.medium.refractiveIndex = 1.62;
	greenGlass.medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(1.0, 0.01, 1.0);

	Material marble;
	SET_DEFAULT_MATERIAL_PROPERTIES(marble);
	marble.specularColor = make_float3(1, 1, 1);
	marble.hasTransmission = true;
	marble.medium.refractiveIndex = 1.486;
	marble.medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(0.6, 0.6, 0.6);
	marble.medium.absorptionAndScatteringProperties.reducedScatteringCoefficient = 8.0;

	Material something;
	SET_DEFAULT_MATERIAL_PROPERTIES(something);
	something.specularColor = make_float3(1, 1, 1);
	something.hasTransmission = true;
	something.medium.refractiveIndex = 1.333;
	something.medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(0.9, 0.3, 0.02);
	something.medium.absorptionAndScatteringProperties.reducedScatteringCoefficient = 2.0;

	Material ketchup;
	SET_DEFAULT_MATERIAL_PROPERTIES(ketchup);
	ketchup.specularColor = make_float3(1, 1, 1);
	ketchup.hasTransmission = true;
	ketchup.medium.refractiveIndex = 1.350;
	ketchup.medium.absorptionAndScatteringProperties.absorptionCoefficient = make_float3(0.02, 5.1, 5.7);
	ketchup.medium.absorptionAndScatteringProperties.reducedScatteringCoefficient = 9.0;

	Material white;
	SET_DEFAULT_MATERIAL_PROPERTIES(white);
	white.diffuseColor = make_float3(0.9, 0.9, 0.9);
	white.emittedColor = make_float3(0, 0, 0);

	Material lightBlue;
	SET_DEFAULT_MATERIAL_PROPERTIES(lightBlue);
	lightBlue.diffuseColor = make_float3(0.4, 0.6, 0.8);
	lightBlue.emittedColor = make_float3(0, 0, 0);
	lightBlue.specularColor = make_float3(1, 1, 1);
	lightBlue.medium.refractiveIndex = 1.2; // Less than water.

	Material gold;
	SET_DEFAULT_MATERIAL_PROPERTIES(gold);
	gold.diffuseColor = make_float3(0, 0, 0);
	gold.emittedColor = make_float3(0, 0, 0);
	gold.specularColor = make_float3(0.869, 0.621, 0.027);
	gold.medium.refractiveIndex = 1000.0; // TODO: Make metal option or something!

	Material steel;
	SET_DEFAULT_MATERIAL_PROPERTIES(steel);
	steel.diffuseColor = make_float3(0, 0, 0);
	steel.emittedColor = make_float3(0, 0, 0);
	steel.specularColor = make_float3(0.89, 0.89, 0.89);
	steel.medium.refractiveIndex = 1000.0; // TODO: Make metal option or something!

	Material light;
	SET_DEFAULT_MATERIAL_PROPERTIES(light);
	light.diffuseColor = make_float3(0, 0, 0);
	light.emittedColor = make_float3(13, 13, 11);

	tempSpheres[0].position = make_float3(-0.9, 0, -0.9);
	tempSpheres[0].radius = 0.8;
	tempSpheres[0].material = steel;

	tempSpheres[1].position = make_float3(0.9, -0.5, 1.3);
	tempSpheres[1].radius = 0.3;
	tempSpheres[1].material = greenGlass;

	tempSpheres[2].position = make_float3(-0.5, -0.4, 1.0);
	tempSpheres[2].radius = 0.4;
	tempSpheres[2].material = lightBlue;

	tempSpheres[3].position = make_float3(-1.0, -0.7, 1.2);
	tempSpheres[3].radius = 0.1;
	tempSpheres[3].material = lightBlue;

	tempSpheres[4].position = make_float3(-0.5, -0.7, 1.7);
	tempSpheres[4].radius = 0.1;
	tempSpheres[4].material = lightBlue;

	tempSpheres[5].position = make_float3(0.3, -0.7, 1.4);
	tempSpheres[5].radius = 0.1;
	tempSpheres[5].material = lightBlue;

	tempSpheres[6].position = make_float3(-0.1, -0.7, 0.1);
	tempSpheres[6].radius = 0.1;
	tempSpheres[6].material = lightBlue;

	tempSpheres[7].position = make_float3(0.2, -0.55, 0.7);
	tempSpheres[7].radius = 0.25;
	tempSpheres[7].material = lightBlue;

	tempSpheres[8].position = make_float3(0.8, 0, -0.4);
	tempSpheres[8].radius = 0.8;
	tempSpheres[8].material = green;

	tempSpheres[9].position = make_float3(0.8, 1.2, -0.4);
	tempSpheres[9].radius = 0.4;
	tempSpheres[9].material = green;

	tempSpheres[10].position = make_float3(0.8, 1.8, -0.4);
	tempSpheres[10].radius = 0.2;
	tempSpheres[10].material = green;

	tempSpheres[11].position = make_float3(0.8, 2.1, -0.4);
	tempSpheres[11].radius = 0.1;
	tempSpheres[11].material = green;

	tempSpheres[12].position = make_float3(0.8, 2.25, -0.4);
	tempSpheres[12].radius = 0.05;
	tempSpheres[12].material = green;

	tempSpheres[13].position = make_float3(0.8, 2.325, -0.4);
	tempSpheres[13].radius = 0.025;
	tempSpheres[13].material = green;



	for(int i = 0; i<numPolys; i++){
		polys[i].position = make_float3(0.0,-0.8,0.0);
		polys[i].material = glass;
	}

    // Copy to GPU:
	CUDA_SAFE_CALL( cudaMalloc( (void**)&spheres, numSpheres * sizeof(Sphere) ) );
	CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_polys, numPolys * sizeof(Poly) ) );
    CUDA_SAFE_CALL( cudaMemcpy( spheres, tempSpheres, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( dev_polys, polys, numPolys * sizeof(Poly), cudaMemcpyHostToDevice) );

    delete [] tempSpheres;
}

void PathTracer::deleteDeviceData() {
    CUDA_SAFE_CALL( cudaFree( spheres ) );

}


