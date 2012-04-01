#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "driver_functions.h"
#include "sm_13_double_functions.h"


#include "cutil_math.h"
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cassert>
#include <ctime>
#include <iostream>
#include <iomanip>
#include <limits>

#include "basic_math.h"

#include "image.h"
#include "sphere.h"
#include "ray.h"
#include "camera.h"

#include "cuda_safe_call.h"

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>

// Settings:
#define BLOCK_SIZE 256 // Number of threads in a block.
#define MAX_TRACE_DEPTH 15 // TODO: Put settings somewhere else and don't make them defines.
#define RAY_BIAS_DISTANCE 0.0002 // TODO: Put with other settings somewhere.

// Numeric constants, copied from BasicMath:
#define PI                    3.1415926535897932384626422832795028841971
#define TWO_PI				  6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476



/*
__device__ float floatInfinity() {
	const unsigned long long ieee754inf = 0x7f800000; // Change the 7 to an f for negative infinity.
	return __longlong_as_float(ieee754inf);
}

__device__ double doubleInfinity() {
	const unsigned long long ieee754inf = 0x7ff0000000000000; // Change the 7 to an f for negative infinity.
	return __longlong_as_double(ieee754inf);
}
*/

__host__ __device__
unsigned int hash(unsigned int a) {
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//E is eye, C is view, U is up
__global__ void raycast_from_camera_kernel(float3 E, float3 C, float3 U, float2 fov, float2 resolution, int numPixels, Ray* rays) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	//get x and y coordinates of pixel
	int y = int(pixelIndex/resolution.y);
	int x = pixelIndex - (y*resolution.y);
	
	if (validIndex) {
		//more compact version, in theory uses fewer registers but horrendously unreadable
		// Now treating FOV as the full FOV, not half, so I multiplied it by 0.5, although I could be missing something.
		// Another optimization we can make is storing the angles in radians.
		float3 PmE = (E+C) + (((2*(x/(resolution.x-1)))-1)*((cross(C,U)*float(length(C)*tan(fov.x*0.5*(PI/180))))/float(length(cross(C,U))))) + (((2*(y/(resolution.y-1)))-1)*((cross(cross(C,U), C)*float(length(C)*tan(-fov.y*0.5*(PI/180))))/float(length(cross(cross(C,U), C))))) - E;
		rays[pixelIndex].origin = E;
		rays[pixelIndex].direction = normalize(PmE);// normalize(E + (float(200)*(PmE))/float(length(PmE)));

		// I wonder how much slower the more legible version actually is. I would lean towards writing clean code before doing optimizations that destroy readability, but it's seems that's not always the point of GPU programming.
		// Also, we can further improve the more legible version with descriptive variable names.

		//more legible version
		/*float CD = length(C);

		float3 A = cross(C,U);
		float3 B = cross(A,C);
		float3 M = E+C;
		float3 H = (A*float(CD*tan(fov.x*0.5*(PI/180))))/float(length(A)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5, although I could be missing something.
		float3 V = (B*float(CD*tan(-fov.y*0.5*(PI/180))))/float(length(B)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5, although I could be missing something.
		
		float sx = x/(resolution.x-1);
		float sy = y/(resolution.y-1);

		float3 P = M + (((2*sx)-1)*H) + (((2*sy)-1)*V);
		float3 PmE = P-E;

		rays[pixelIndex].direction = normalize(PmE); // normalize(E + (float(200)*(PmE))/float(length(PmE))); // The E + and the 200 weren't necessary.
		rays[pixelIndex].origin = E;*/

		//accumulatedColors[pixelIndex] = rays[pixelIndex].direction;	//test code, should output green/yellow/black/red if correct
	}
}



__host__ __device__
float findGroundPlaneIntersection(float elevation, const Ray & ray, float3 & intersectionPoint, float3 & normal) {
	// Only finds intersections with the top of the plane.

	if (ray.direction.y < 0) {

		double t = (elevation - ray.origin.y) / ray.direction.y;
	
		intersectionPoint = ray.origin + t * ray.direction;
		normal = make_float3(0, 1, 0);
		
		return t;
	}
	
	return -1; // No intersection.
}



__host__ __device__
//assumes that ray is already transformed into sphere's object space, returns -1 if no intersection
float findSphereIntersection(const Sphere & sphere, const Ray & ray, float3 & intersectionPoint, float3 & normal) {




	// Based on math at http://en.wikipedia.org/wiki/Ray_tracing_%28graphics%29

	float3 v = ray.origin - sphere.position; // Sphere position relative to ray origin.
	float vDotDirection = dot(v, ray.direction);
	float radicand = vDotDirection * vDotDirection - (dot(v, v) - sphere.radius * sphere.radius);
	if (radicand < 0) return -1;
	float squareRoot = sqrt(radicand);
	float firstTerm = -vDotDirection;
	float t1 = firstTerm + squareRoot;
	float t2 = firstTerm - squareRoot;

	float t;

	if (t1 < 0 && t2 < 0) { // (t1 < 0.01 && t2 < 0.01) { // Epsilon shouldn't be necessary here if we have a good global ray bias system.
		return -1;
	} else if (t1 > 0 && t2 > 0) { // (t1 >= 0.01 && t2 >= 0.01) { // Epsilon shouldn't be necessary here if we have a good global ray bias system.
		t = min(t1, t2);
	} else {
		t = max(t1, t2);
	}

	intersectionPoint = ray.origin + t * ray.direction;
	normal = normalize(intersectionPoint - sphere.position);
	return t;

















	/*
	// TEST:
	if (sqrt(ray.direction.x * ray.direction.x + ray.direction.y * ray.direction.y) < 0.1) { //if (ray.direction.z < 0 && ray.direction.y < 0 && ray.direction.z < 0) {
		intersectionPoint = ray.origin + 5.0*ray.direction;
		normal = make_float3(0, 0, -1); //normalize(intersectionPoint - sphere.position);
		return 5.0;
	} else {
		return -1;
	}
	*/














	/*
	// http://en.wikipedia.org/wiki/Discriminant
	// http://mathworld.wolfram.com/QuadraticFormula.html
	// http://en.wikipedia.org/wiki/Ray_tracing_%28graphics%29

	normal = make_float3(0,0,0);

	Ray transformedRay;
	transformedRay.origin = ray.origin - sphere.position;
	transformedRay.direction = ray.direction;

	float A = dot(transformedRay.direction, transformedRay.direction);
	float B = 2.0f*dot(transformedRay.direction, transformedRay.origin);
	float C = dot(transformedRay.direction, transformedRay.origin) - (sphere.radius*sphere.radius);

	float discriminant = (B*B)-(4*A*C);
	if(discriminant<0){
		return -1;
	}

	float discriminantSqrt = sqrtf(discriminant);
	float q;
	if(B<0){
        q = (-B - discriminantSqrt) * 0.5; // Changed from / 2.0 to * 0.5 for slightly better performance, although maybe the compiler would do this automatically.
    }else{
        q = (-B + discriminantSqrt) * 0.5;
	}

	
	float t0 = q/A;
    float t1 = C/q;

	// Make t0 the first intersection distance along the ray, and t1 the second:
	if(t0>t1){
		// Swap t0 and t1:
		float temp = t0;
		t0 = t1;
		t1 = temp;
    }

	if(t1<0){
		// Both distances are negative. 
		return -1;
    }

	if(t0<0){
		intersectionPoint = ray.origin + t1*ray.direction;
		normal = normalize(intersectionPoint - sphere.position);
		return t1;
	}else{
		intersectionPoint = ray.origin + t0*ray.direction;
		normal = normalize(intersectionPoint - sphere.position);
		return t0;
	}
	*/
}

__host__ __device__
float3 cosineWeightedDirectionInHemisphere(const float3 & normal, float xi1, float xi2) {

    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1.0 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;

	// Find any two perpendicular directions:
	// Either all of the components of the normal are equal to the square root of one third, or at least one of the components of the normal is less than the square root of 1/3.
	float3 directionNotNormal;
	if (abs(normal.x) < SQRT_OF_ONE_THIRD) { 
		directionNotNormal = make_float3(1, 0, 0);
	} else if (abs(normal.y) < SQRT_OF_ONE_THIRD) { 
		directionNotNormal = make_float3(0, 1, 0);
	} else {
		directionNotNormal = make_float3(0, 0, 1);
	}
	float3 perpendicular1 = normalize( cross(normal, directionNotNormal) );
	float3 perpendicular2 =            cross(normal, perpendicular1); // Normalized by default.
  
    return ( up * normal ) + ( cos(around) * over * perpendicular1 ) + ( sin(around) * over * perpendicular2 );

}

__global__ void trace_ray_kernel(int numSpheres, Sphere* spheres, int numPixels, Ray* rays, int rayDepth, float3* notAbsorbedColors, float3* accumulatedColors, unsigned long seed) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	thrust::default_random_engine rng( hash(seed) * hash(pixelIndex) * hash(rayDepth) );
	thrust::uniform_real_distribution<float> uniformDistribution(0,1);

	if (validIndex) {

		// TODO: Restructure this block! It's a mess. I want polymorphism!

		// Reusables:
		float t;
		float3 intersectionPoint;
		float3 normal;

		float bestT = 123456789; // floatInfinity(); // std::numeric_limits<float>::infinity();
		float3 bestIntersectionPoint;// = make_float3(0,0,0);
		float3 bestNormal;// = make_float3(0,0,0);
		bool bestIsGroundPlane = false;
		bool bestIsSphere = false;
		int bestSphereIndex = -1;

		// Check for ground plane intersection:
		float hardCodedGroundPlaneElevation = -0.8; // TODO: Put with other settings somewhere.
		t = findGroundPlaneIntersection(hardCodedGroundPlaneElevation, rays[pixelIndex], intersectionPoint, normal); // 123456789; // floatInfinity(); // std::numeric_limits<float>::infinity();
		if (t > 0) { // No "<" conditional only because this is being tested before anythign else.
			bestT = t;
			bestIntersectionPoint = intersectionPoint;
			bestNormal = normal;

			bestIsGroundPlane = true;
			bestIsSphere = false;
		}
		
		// Check for sphere intersection:
		for (int i = 0; i < numSpheres; i++) {
			t = findSphereIntersection(spheres[i], rays[pixelIndex], intersectionPoint, normal);
			if (t > 0 && t < bestT) {
				bestT = t;
				bestIntersectionPoint = intersectionPoint;
				bestNormal = normal;

				bestIsGroundPlane = false;
				bestIsSphere = true;

				bestSphereIndex = i;
			}
		}

		if (bestIsGroundPlane || bestIsSphere) {

			if (bestIsGroundPlane) {
				float3 hardCodedGroundPlaneDiffuseColor = make_float3(0.455, 0.43, 0.39);
				//accumulatedColors[pixelIndex] += NOTHING;
				notAbsorbedColors[pixelIndex] *= hardCodedGroundPlaneDiffuseColor;
			} else if (bestIsSphere) {
				accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * spheres[bestSphereIndex].emittedColor;
				notAbsorbedColors[pixelIndex] *= spheres[bestSphereIndex].diffuseColor;
			}

			// TODO: Use Russian roulette instead of simple multipliers!
				
			// Choose a new ray direction:
			float randomFloat1 = uniformDistribution(rng); 
			float randomFloat2 = uniformDistribution(rng); 
			float3 newRayDirection = cosineWeightedDirectionInHemisphere(bestNormal, randomFloat1, randomFloat2);
			rays[pixelIndex].origin = bestIntersectionPoint + ( RAY_BIAS_DISTANCE * bestNormal ); // TODO: Bias ray in the other direction if the new ray is transmitted.
			rays[pixelIndex].direction = newRayDirection;

		} else {
			float3 hardCodedBackgroundColor = make_float3(0.15, 0.25, 0.4);
			accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * hardCodedBackgroundColor;
			notAbsorbedColors[pixelIndex] = make_float3(0,0,0); // The ray now has zero weight. TODO: Terminate the ray.
		}



		/*
		// TEST:
		// Generate a random number:
		// TODO: Generate more random numbers at a time to speed this up significantly!
		float randomFloat = uniformDistribution(rng); 

		if (randomFloat < 0.5) {
			accumulatedColors[pixelIndex] = rays[pixelIndex].direction;
		} else {
			accumulatedColors[pixelIndex] = make_float3(0,0,0);
		}
		*/


	}

}

extern "C"
void launch_kernel(int numSpheres, Sphere* spheres, int numPixels, Color* pixels, Ray* rays, int counter, Camera* rendercam) {
	
	// Configure grid and block sizes:
	int threadsPerBlock = BLOCK_SIZE;
	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int blocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;


	// Initialize color arrays:
	Color* tempNotAbsorbedColors = (Color*)malloc(numPixels * sizeof(Color));
	Color* tempAccumulatedColors = (Color*)malloc(numPixels * sizeof(Color));
	for (int i = 0; i < numPixels; i++) {
		tempNotAbsorbedColors[i] = make_float3(1,1,1);
		tempAccumulatedColors[i] = make_float3(0,0,0);
	}
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void**)&notAbsorbedColors, numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&accumulatedColors, numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMemcpy( notAbsorbedColors, tempNotAbsorbedColors, numPixels * sizeof(Color), cudaMemcpyHostToDevice) );
	CUDA_SAFE_CALL( cudaMemcpy( accumulatedColors, tempAccumulatedColors, numPixels * sizeof(Color), cudaMemcpyHostToDevice) );
	free(tempNotAbsorbedColors);
	free(tempAccumulatedColors);


	//only launch raycast from camera kernel if this is the first ever pass!
	// I think we'll have to run this every pass if we want to do anti-aliasing using jittering.
	// Also, if we don't want to re-compute the camera rays, we'll need a separate array for secondary rays.
	//if (counter == 0) {
	raycast_from_camera_kernel<<<blocksPerGrid, threadsPerBlock>>>(rendercam->position, rendercam->view, rendercam->up, rendercam->fov, rendercam->resolution, numPixels, rays);
	//}


	for (int rayDepth = 0; rayDepth < MAX_TRACE_DEPTH; rayDepth++) {
		trace_ray_kernel<<<blocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, numPixels, rays, rayDepth, notAbsorbedColors, accumulatedColors, counter);
	}



	// Copy the accumulated colors from the device into the host image:
	CUDA_SAFE_CALL( cudaMemcpy( pixels, accumulatedColors, numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );


	// Clean up:
	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );

}
