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
#include "fresnel.h"
#include "medium.h"
#include "absorption_and_scattering_properties.h"

#include "cuda_safe_call.h"

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>



// Settings:
#define BLOCK_SIZE 256 // Number of threads in a block.
#define MAX_TRACE_DEPTH 40 // TODO: Put settings somewhere else and don't make them defines.
#define RAY_BIAS_DISTANCE 0.0002 // TODO: Put with other settings somewhere.
#define MIN_RAY_WEIGHT 0.00001 // Terminate rays below this weight.
#define HARD_CODED_GROUND_ELEVATION -0.8


// Numeric constants, copied from BasicMath:
#define PI                    3.1415926535897932384626422832795028841971
#define TWO_PI				  6.2831853071795864769252867665590057683943
#define SQRT_OF_ONE_THIRD     0.5773502691896257645091487805019574556476
#define E                     2.7182818284590452353602874713526624977572





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

__global__ void initializeThings(int numPixels, int* activePixels, AbsorptionAndScatteringProperties* absorptionAndScattering, Color* notAbsorbedColors, Color* accumulatedColors) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		activePixels[pixelIndex] = pixelIndex;
		SET_TO_AIR_ABSORPTION_AND_SCATTERING_PROPERTIES(absorptionAndScattering[pixelIndex]);
		notAbsorbedColors[pixelIndex] = make_float3(1,1,1);
		accumulatedColors[pixelIndex] = make_float3(0,0,0);

	}

}
/*
__global__ void countActivePixels(int numActivePixels, Color* notAbsorbedColors, int* count) {
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int activePixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (activePixelIndex < numActivePixels);

	if (validIndex) {

		if (notAbsorbedColors[activePixels[activePixelIndex]] > 0) {
			atomicAdd(count, 1); // TODO: Do a parallel reduction instead!
		}

	}

}

void resizeActivePixels(int numActivePixels, int* activePixels, int* newActivePixels) {

}
*/

__global__ void raycastFromCameraKernel(float3 eye, float3 view, float3 up, float2 fov, float2 resolution, int numPixels, Ray* rays, unsigned long seed) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		//get x and y coordinates of pixel
		int y = int(pixelIndex/resolution.y);
		int x = pixelIndex - (y*resolution.y);
	
		//generate random jitter offsets for supersampled antialiasing
		thrust::default_random_engine rng( hash(seed) * hash(pixelIndex) * hash(seed) );
		thrust::uniform_real_distribution<float> uniformDistribution(-0.5, 0.5);
		float jitterValueX = uniformDistribution(rng); 
		float jitterValueY = uniformDistribution(rng); 


		float lengthOfView = length(view);

		float3 A = cross(view, up);
		float3 B = cross(A, view);
		float3 middle = eye + view;
		float3 horizontal = ( A * float(lengthOfView * tan(fov.x * 0.5 * (PI/180)))) / float(length(A)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5, although I could be missing something.
		float3 vertical = ( B * float(lengthOfView * tan(-fov.y * 0.5 * (PI/180)))) / float(length(B)); // Now treating FOV as the full FOV, not half, so I multiplied it by 0.5, although I could be missing something.
		
		float sx = (jitterValueX+x)/(resolution.x-1);
		float sy = (jitterValueY+y)/(resolution.y-1);

		float3 point = middle + (((2*sx)-1)*horizontal) + (((2*sy)-1)*vertical);
		float3 eyeToPoint = point - eye;

		rays[pixelIndex].direction = normalize(eyeToPoint);
		rays[pixelIndex].origin = eye;

		//accumulatedColors[pixelIndex] = rays[pixelIndex].direction;	//test code, should output green/yellow/black/red if correct
	}
}


__host__ __device__
float3 positionAlongRay(const Ray & ray, float t) {
	return ray.origin + t * ray.direction;
}


__host__ __device__
float findGroundPlaneIntersection(float elevation, const Ray & ray, float3 & normal) {

	if (ray.direction.y != 0) {

		double t = (elevation - ray.origin.y) / ray.direction.y;

		if (ray.direction.y < 0) { // Top of plane.
			normal = make_float3(0, 1, 0);
		} else { // Bottom of plane.
			normal = make_float3(0, 1, 0);//make_float3(0, -1, 0); // Make the normal negative for opaque appearance. Positive normal lets you see diffusely through the ground which looks really cool and gives you a better idea of where you are!
		}
		
		return t;

	}

	
	return -1; // No intersection.
}



__host__ __device__
//assumes that ray is already transformed into sphere's object space, returns -1 if no intersection
float findSphereIntersection(const Sphere & sphere, const Ray & ray, float3 & normal) {



	// Copied from Photorealizer and modified.
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

	float3 intersectionPoint = positionAlongRay(ray, t);
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
float3 randomCosineWeightedDirectionInHemisphere(const float3 & normal, float xi1, float xi2) {

    float up = sqrt(xi1); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
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

__host__ __device__
float3 randomDirectionInSphere(float xi1, float xi2) {

    float up = xi1 * 2 - 1; // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = xi2 * TWO_PI;

    return make_float3( up, cos(around) * over, sin(around) * over );

}

__host__ __device__
float3 computeReflectionDirection(const float3 & normal, const float3 & incident) {
	return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__
float3 computeTransmissionDirection(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted) {
	// Snell's Law:
	// Copied from Photorealizer.

	float cosTheta1 = dot(normal, incident);

	float n1_n2 =  refractiveIndexIncident /  refractiveIndexTransmitted;

	float radicand = 1 - pow(n1_n2, 2) * (1 - pow(cosTheta1, 2));
	if (radicand < 0) return make_float3(0, 0, 0); // Return value???????????????????????????????????????
	float cosTheta2 = sqrt(radicand);

	if (cosTheta1 > 0) { // normal and incident are on same side of the surface.
		return n1_n2 * (-1 * incident) + ( n1_n2 * cosTheta1 - cosTheta2 ) * normal;
	} else { // normal and incident are on opposite sides of the surface.
		return n1_n2 * (-1 * incident) + ( n1_n2 * cosTheta1 + cosTheta2 ) * normal;
	}

}

__host__ __device__
Fresnel computeFresnel(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted, const float3 & reflectionDirection, const float3 & transmissionDirection) {
	Fresnel fresnel;


	
	// First, check for total internal reflection:
	if ( length(transmissionDirection) <= 0.12345 || dot(normal, transmissionDirection) > 0 ) { // The length == 0 thing is how we're handling TIR right now.
		// Total internal reflection!
		fresnel.reflectionCoefficient = 1;
		fresnel.transmissionCoefficient = 0;
		return fresnel;
	}



	// Real Fresnel equations:
	// Copied from Photorealizer.
	float cosThetaIncident = dot(normal, incident);
	float cosThetaTransmitted = dot(-1 * normal, transmissionDirection);
	float reflectionCoefficientSPolarized = pow(   (refractiveIndexIncident * cosThetaIncident - refractiveIndexTransmitted * cosThetaTransmitted)   /   (refractiveIndexIncident * cosThetaIncident + refractiveIndexTransmitted * cosThetaTransmitted)   , 2);
    float reflectionCoefficientPPolarized = pow(   (refractiveIndexIncident * cosThetaTransmitted - refractiveIndexTransmitted * cosThetaIncident)   /   (refractiveIndexIncident * cosThetaTransmitted + refractiveIndexTransmitted * cosThetaIncident)   , 2);
	float reflectionCoefficientUnpolarized = (reflectionCoefficientSPolarized + reflectionCoefficientPPolarized) / 2.0; // Equal mix.
	//
	fresnel.reflectionCoefficient = reflectionCoefficientUnpolarized;
	fresnel.transmissionCoefficient = 1 - fresnel.reflectionCoefficient;
	return fresnel;
	
	/*
	// Shlick's approximation including expression for R0 and modification for transmission found at http://www.bramz.net/data/writings/reflection_transmission.pdf
	// TODO: IMPLEMENT ACTUAL FRESNEL EQUATIONS!
	float R0 = pow( (refractiveIndexIncident - refractiveIndexTransmitted) / (refractiveIndexIncident + refractiveIndexTransmitted), 2 ); // For Schlick's approximation.
	float cosTheta;
	if (refractiveIndexIncident <= refractiveIndexTransmitted) {
		cosTheta = dot(normal, incident);
	} else {
		cosTheta = dot(-1 * normal, transmissionDirection); // ???
	}
	fresnel.reflectionCoefficient = R0 + (1.0 - R0) * pow(1.0 - cosTheta, 5); // Costly pow function might make this slower than actual Fresnel equations. TODO: USE ACTUAL FRESNEL EQUATIONS!
	fresnel.transmissionCoefficient = 1.0 - fresnel.reflectionCoefficient;
	return fresnel;
	*/

}

__host__ __device__
Color computeBackgroundColor(const float3 & direction) {
	float position = (dot(direction, normalize(make_float3(-0.5, 0.5, -1.0))) + 1) / 2;
	Color firstColor = make_float3(0.15, 0.3, 0.5); // Bluish.
	Color secondColor = make_float3(1.0, 1.0, 1.0); // White.
	Color interpolatedColor = (1 - position) * firstColor + position * secondColor;
	float radianceMultiplier = 0.3;
	return interpolatedColor * radianceMultiplier;
}

__host__ __device__
Color computeTransmission(Color absorptionCoefficient, float distance) {
	Color transmitted;
	transmitted.x = pow((float)E, (float)(-1 * absorptionCoefficient.x * distance));
	transmitted.y = pow((float)E, (float)(-1 * absorptionCoefficient.y * distance));
	transmitted.z = pow((float)E, (float)(-1 * absorptionCoefficient.z * distance));
	return transmitted;
}

__global__ void traceRayKernel(int numSpheres, Sphere* spheres, int numActivePixels, int* activePixels, Ray* rays, int rayDepth, AbsorptionAndScatteringProperties* absorptionAndScattering, float3* notAbsorbedColors, float3* accumulatedColors, unsigned long seedOrPass) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int activePixelIndex = BLOCK_SIZE * bx + tx;
	if (activePixelIndex >= numActivePixels) return;

	// TODO: Restructure stuff! It's a mess. Use classes!

	int pixelIndex = activePixels[activePixelIndex];
	Ray currentRay = rays[pixelIndex];

	thrust::default_random_engine rng( hash(seedOrPass) * hash(pixelIndex) * hash(rayDepth) );
	thrust::uniform_real_distribution<float> uniformDistribution(0,1);

	float bestT = 123456789; // floatInfinity(); // std::numeric_limits<float>::infinity();
	float3 bestNormal;// = make_float3(0,0,0);
	bool bestIsGroundPlane = false;
	bool bestIsSphere = false;
	int bestSphereIndex = -1;

	//if (true) { // Test. I did this to contain the local variables to isolate a bug (I suspected that I was using one of these local variables below by mistake, so scoping them caused a helpful compiler error to tell me where). But now I'm also wondering if those local variables just sit there and waste registers if they don't go out of scope.

	// Reusables:
	float t;
	float3 normal;

	// Check for ground plane intersection:
	t = findGroundPlaneIntersection(HARD_CODED_GROUND_ELEVATION, currentRay, normal); // 123456789; // floatInfinity(); // std::numeric_limits<float>::infinity();
	if (t > 0) { // No "<" conditional only because this is being tested before anything else.
		bestT = t;
		bestNormal = normal;

		bestIsGroundPlane = true;
		bestIsSphere = false;
	}
		
	// Check for sphere intersection:
	for (int i = 0; i < numSpheres; i++) {
		t = findSphereIntersection(spheres[i], currentRay, normal);
		if (t > 0 && t < bestT) {
			bestT = t;
			bestNormal = normal;

			bestIsGroundPlane = false;
			bestIsSphere = true;

			bestSphereIndex = i;
		}
	}







	// ABSORPTION AND SCATTERING:
	{ // BEGIN SCOPE.
		AbsorptionAndScatteringProperties currentAbsorptionAndScattering = absorptionAndScattering[pixelIndex];
		#define ZERO_ABSORPTION_EPSILON 0.00001
		if ( currentAbsorptionAndScattering.reducedScatteringCoefficient > 0 || dot(currentAbsorptionAndScattering.absorptionCoefficient, currentAbsorptionAndScattering.absorptionCoefficient) > ZERO_ABSORPTION_EPSILON ) { // The dot product with itself is equivalent to the squre of the length.
			float randomFloatForScatteringDistance = uniformDistribution(rng);
			float scatteringDistance = -log(randomFloatForScatteringDistance) / absorptionAndScattering[pixelIndex].reducedScatteringCoefficient;
			if (scatteringDistance < bestT) {
				// Both absorption and scattering.

				// Scatter the ray:
				Ray nextRay;
				nextRay.origin = positionAlongRay(currentRay, scatteringDistance);
				float randomFloatForScatteringDirection1 = uniformDistribution(rng);
				float randomFloatForScatteringDirection2 = uniformDistribution(rng);
				nextRay.direction = randomDirectionInSphere(randomFloatForScatteringDirection1, randomFloatForScatteringDirection2); // Isoptropic scattering!
				rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.

				// Compute how much light was absorbed along the ray before it was scattered:
				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, scatteringDistance);

				// DUPLICATE CODE:
				// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
				if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // TODO: Faster: dot product of a vector with itself is the same as its length squared.
					activePixels[activePixelIndex] = -1;
				}

				// That's it for this iteration!
				return; // IMPORTANT!
			} else {
				// Just absorption.

				notAbsorbedColors[pixelIndex] *= computeTransmission(currentAbsorptionAndScattering.absorptionCoefficient, bestT);

				// Now proceed to compute interaction with intersected object and whatnot!
			}
		}
	} // END SCOPE.






	


	if (bestIsGroundPlane || bestIsSphere) {


		Material bestMaterial;
		if (bestIsGroundPlane) {
			Material hardCodedGroundMaterial;
			SET_DEFAULT_MATERIAL_PROPERTIES(hardCodedGroundMaterial);
			hardCodedGroundMaterial.diffuseColor = make_float3(0.455, 0.43, 0.39);
//			hardCodedGroundMaterial.emittedColor = make_float3(0,0,0);
//			hardCodedGroundMaterial.specularColor = make_float3(0,0,0);
//			hardCodedGroundMaterial.specularRefractiveIndex = 0;
//			hardCodedGroundMaterial.hasTransmission = false;
			bestMaterial = hardCodedGroundMaterial;
		} else if (bestIsSphere) {
			bestMaterial = spheres[bestSphereIndex].material;
		}




		// TODO: Reduce duplicate code and memory usage here and in the functions called here.
		// TODO: Finish implementing the functions called here.
		// TODO: Clean all of this up!
		float3 incident = -currentRay.direction;

		Medium incidentMedium;
		SET_TO_AIR_MEDIUM(incidentMedium);
		Medium transmittedMedium = bestMaterial.medium;

		bool backFace = ( dot(bestNormal, incident) < 0 );

		if (backFace) {
			// Flip the normal:
			bestNormal *= -1;
			// Swap the IORs:
			// TODO: Use the BasicMath swap function if possible on the device.
			Medium tempMedium = incidentMedium;
			incidentMedium = transmittedMedium;
			transmittedMedium = tempMedium;
		}





		float3 reflectionDirection = computeReflectionDirection(bestNormal, incident);
		float3 transmissionDirection = computeTransmissionDirection(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex);

		float3 bestIntersectionPoint = positionAlongRay(currentRay, bestT);

		float3 biasVector = ( RAY_BIAS_DISTANCE * bestNormal ); // TODO: Bias ray in the other direction if the new ray is transmitted!!!

		bool doSpecular = ( bestMaterial.medium.refractiveIndex > 1.0 ); // TODO: Move?
		float rouletteRandomFloat = uniformDistribution(rng);
		// TODO: Fix long conditional, and maybe lots of temporary variables.
		// TODO: Optimize total internal reflection case (no random number necessary in that case).
		bool reflectFromSurface = ( doSpecular && rouletteRandomFloat < computeFresnel(bestNormal, incident, incidentMedium.refractiveIndex, transmittedMedium.refractiveIndex, reflectionDirection, transmissionDirection).reflectionCoefficient );
		if (reflectFromSurface) {
			// Ray reflected from the surface. Trace a ray in the reflection direction.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			notAbsorbedColors[pixelIndex] *= bestMaterial.specularColor;

			Ray nextRay;
			nextRay.origin = bestIntersectionPoint + biasVector;
			nextRay.direction = reflectionDirection;
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		} else if (bestMaterial.hasTransmission) {
			// Ray transmitted and refracted.

			// The ray has passed into a new medium!
			absorptionAndScattering[pixelIndex] = transmittedMedium.absorptionAndScatteringProperties;

			Ray nextRay;
			nextRay.origin = bestIntersectionPoint - biasVector; // Bias ray in the other direction because it's transmitted!!!
			nextRay.direction = transmissionDirection;
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		} else {
			// Ray did not reflect from the surface, so consider emission and take a diffuse sample.

			// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
			accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * bestMaterial.emittedColor;
			notAbsorbedColors[pixelIndex] *= bestMaterial.diffuseColor;

			// Choose a new ray direction:
			float randomFloat1 = uniformDistribution(rng); 
			float randomFloat2 = uniformDistribution(rng); 
			Ray nextRay;
			nextRay.origin = bestIntersectionPoint + biasVector;
			nextRay.direction = randomCosineWeightedDirectionInHemisphere(bestNormal, randomFloat1, randomFloat2);
			rays[pixelIndex] = nextRay; // Only assigning to global memory ray once, for better performance.
		}


		// DUPLICATE CODE:
		// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
		if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // TODO: Faster: dot product of a vector with itself is the same as its length squared.
			activePixels[activePixelIndex] = -1;
		}

	} else {
		// Ray didn't hit an object, so sample the background and terminate the ray.

		accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * computeBackgroundColor(currentRay.direction);
		//notAbsorbedColors[pixelIndex] = make_float3(0,0,0); // The ray now has zero weight. // TODO: Remove this? This isn't even necessary because we know the ray will be terminated anyway.

		activePixels[activePixelIndex] = -1; // To assist Thrust stream compaction, set this activePixel to -1 because the ray weight is now zero.
	}

	// MOVED:
	//// To assist Thrust stream compaction, set this activePixel to -1 if the ray weight is now zero:
	//if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // Faster: dot product of a vector with itself is the same as its length squared.
	//	activePixels[activePixelIndex] = -1;
	//}


}



// http://docs.thrust.googlecode.com/hg/group__counting.html
// http://docs.thrust.googlecode.com/hg/group__stream__compaction.html
struct isNegative
{
	__host__ __device__ 
	bool operator()(const int & x) 
	{
		return x < 0;
	}
};



extern "C"
void launchKernel(int numSpheres, Sphere* spheres, int numPixels, Color* pixels, int counter, Camera* renderCam) {
	


	// Configure grid and block sizes:
	int threadsPerBlock = BLOCK_SIZE;

	// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
	int fullBlocksPerGrid = (numPixels + threadsPerBlock - 1) / threadsPerBlock;





	// Declare and allocate active pixels, color arrays, and rays:
	int* activePixels = NULL;
	Ray* rays = NULL;
	AbsorptionAndScatteringProperties* absorptionAndScattering = NULL;
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void**)&activePixels, numPixels * sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&rays, numPixels * sizeof(Ray)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&absorptionAndScattering, numPixels * sizeof(AbsorptionAndScatteringProperties)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&notAbsorbedColors, numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&accumulatedColors, numPixels * sizeof(Color)) );

	initializeThings<<<fullBlocksPerGrid, threadsPerBlock>>>(numPixels, activePixels, absorptionAndScattering, notAbsorbedColors, accumulatedColors);

	int numActivePixels = numPixels;

	// Run this every pass so we can do anti-aliasing using jittering.
	// If we don't want to re-compute the camera rays, we'll need a separate array for secondary rays.
	
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->position, renderCam->view, renderCam->up, renderCam->fov, renderCam->resolution, numPixels, rays, counter);





	for (int rayDepth = 0; rayDepth < MAX_TRACE_DEPTH; rayDepth++) {

		// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
		int newBlocksPerGrid = (numActivePixels + threadsPerBlock - 1) / threadsPerBlock; // Duplicate code.

		traceRayKernel<<<newBlocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, numActivePixels, activePixels, rays, rayDepth, absorptionAndScattering, notAbsorbedColors, accumulatedColors, counter);
		


		// Use Thrust stream compaction to compress activePixels:
		// http://docs.thrust.googlecode.com/hg/group__stream__compaction.html#ga5fa8f86717696de88ab484410b43829b
		thrust::device_ptr<int> devicePointer(activePixels);
		thrust::device_ptr<int> newEnd = thrust::remove_if(devicePointer, devicePointer + numActivePixels, isNegative());
		numActivePixels = newEnd.get() - activePixels;
		//std::cout << numActivePixels << std::endl;
		//break;



		/*
		// Compress activePixels:
		// SUPER SLOW.
		// TODO: Use custom kernels or thrust to avoid memory copying!!!!!!!!!!!!!!!
		// TODO: Use parallel reduction for counting, and stream compaction!!!!!!!!!!!!!!!!!!!!
		// Middle ground alternative: maybe just keep a device array of booleans to mark active pixels, then use that here instead of the heavier notAbsorbedColors.
		int* tempActivePixels = new int[numActivePixels];
		Color* tempNotAbsorbedColors = new Color[numPixels];;
		CUDA_SAFE_CALL( cudaMemcpy( tempActivePixels, activePixels, numActivePixels * sizeof(int), cudaMemcpyDeviceToHost) );
		CUDA_SAFE_CALL( cudaMemcpy( tempNotAbsorbedColors, notAbsorbedColors, numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );
		int livingCount = 0;
		for (int i = 0; i < numActivePixels; i++) {
			if (length(tempNotAbsorbedColors[tempActivePixels[i]]) > MIN_RAY_WEIGHT) { // TODO: Use a length_squared function for better performance!!!!!!
				livingCount++;
			}
		}
		int* tempCompressedActivePixels = new int[livingCount];
		int currentIndex = 0;
		for (int i = 0; i < numActivePixels; i++) { // Duplicate code. 
			if (length(tempNotAbsorbedColors[tempActivePixels[i]]) > MIN_RAY_WEIGHT) { // Should be true for exactly the same elements as above.
				tempCompressedActivePixels[currentIndex] = tempActivePixels[i];
				currentIndex++;
			}
		}
		CUDA_SAFE_CALL( cudaFree( activePixels ) );
		CUDA_SAFE_CALL( cudaMalloc((void**)&activePixels, livingCount * sizeof(int)) );
		CUDA_SAFE_CALL( cudaMemcpy( activePixels, tempCompressedActivePixels, livingCount * sizeof(int), cudaMemcpyHostToDevice) );
		numActivePixels = livingCount; // Could alternatively do this earlier.
		//std::cout << livingCount << std::endl; // TEST.
		delete [] tempActivePixels;
		delete [] tempNotAbsorbedColors;
		delete [] tempCompressedActivePixels;
		*/
		/*
		int numNewActivePixels = 0;
		countActivePixels(numActivePixels, notAbsorbedColors, newNumActivePixels);
		int* oldActivePixels = activePixels;
		int* newActivePixels = NULL;
		CUDA_SAFE_CALL( cudaMalloc((void**)&newActivePixels, numNewActivePixels * sizeof(int)) );
		refillActivePixels(numActivePixels, oldActivePixels, newActivePixels);
		CUDA_SAFE_CALL( cudaFree( oldActivePixels ) );
		*/


	}




	// Copy the accumulated colors from the device into the host image:
	CUDA_SAFE_CALL( cudaMemcpy( pixels, accumulatedColors, numPixels * sizeof(Color), cudaMemcpyDeviceToHost) );



	// Clean up:
	// TODO: Save these things for the next iteration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
	CUDA_SAFE_CALL( cudaFree( activePixels ) );
	CUDA_SAFE_CALL( cudaFree( rays ) );
	CUDA_SAFE_CALL( cudaFree( absorptionAndScattering ) );
	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );


}
