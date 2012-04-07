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

#include "cuda_safe_call.h"

#include <thrust/random.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <thrust/remove.h>



// Settings:
#define BLOCK_SIZE 256 // Number of threads in a block.
#define MAX_TRACE_DEPTH 11 // TODO: Put settings somewhere else and don't make them defines.
#define RAY_BIAS_DISTANCE 0.0002 // TODO: Put with other settings somewhere.
#define MIN_RAY_WEIGHT 0.00001 // Terminate rays below this weight.
#define AIR_IOR 1.000293 // Don't put this here!

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

__global__ void initializeActivePixelsAndColors(int numPixels, int* activePixels, Color* notAbsorbedColors, Color* accumulatedColors) {

	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int pixelIndex = BLOCK_SIZE * bx + tx;
	bool validIndex = (pixelIndex < numPixels);

	if (validIndex) {

		activePixels[pixelIndex] = pixelIndex;
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
float findGroundPlaneIntersection(float elevation, const Ray & ray, float3 & intersectionPoint, float3 & normal) {

	if (ray.direction.y != 0) {

		double t = (elevation - ray.origin.y) / ray.direction.y;
	
		intersectionPoint = ray.origin + t * ray.direction;

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

__host__ __device__
float3 computeReflectionDirection(const float3 & normal, const float3 & incident) {
	return 2.0 * dot(normal, incident) * normal - incident;
}

__host__ __device__
float3 computeTransmissionDirection(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted) {
	return make_float3(0,0,0); // TODO: IMPLEMENT THIS USING SNELL'S LAW!!!
}

__host__ __device__
Fresnel computeFresnel(const float3 & normal, const float3 & incident, float refractiveIndexIncident, float refractiveIndexTransmitted, const float3 & reflectionDirection, const float3 & transmissionDirection) {
	// Shlick's approximation including expression for R0 found at http://www.bramz.net/data/writings/reflection_transmission.pdf
	// TODO: IMPLEMENT ACTUAL FRESNEL EQUATIONS, OR THE FULL SCHLICK'S APPROXIMATION WITH TRANSMISSION (BASED ON THE LINK ABOVE)!!!
	Fresnel fresnel;
	float R0 = pow( (refractiveIndexIncident - refractiveIndexTransmitted) / (refractiveIndexIncident + refractiveIndexTransmitted), 2 ); // For Schlick's approximation.
	fresnel.reflectionCoefficient = R0 + (1.0 - R0) * pow(1.0 - dot(normal, incident), 5.0);
	fresnel.transmissionCoefficient = 1.0 - fresnel.reflectionCoefficient;
	return fresnel;
}

__host__ __device__
float3 computeBackgroundColor(const float3 & direction) {
	float3 darkSkyBlue = make_float3(0.15, 0.25, 0.4); // Dark grayish-blue.
	return darkSkyBlue * ((dot(direction, normalize(make_float3(-0.5, 0.0, -1.0))) + 1 + 1) / 2);
}

__global__ void traceRayKernel(int numSpheres, Sphere* spheres, int numActivePixels, int* activePixels, Ray* rays, int rayDepth, float3* notAbsorbedColors, float3* accumulatedColors, unsigned long seedOrPass) {

//__shared__ float4 something[BLOCK_SIZE]; // 256 (threads per block) * 4 (floats per thread) * 4 (bytes per float) = 4096 (bytes per block)

	// Duplicate code:
	int bx = blockIdx.x;
	int tx = threadIdx.x;
	int activePixelIndex = BLOCK_SIZE * bx + tx;
	bool validActivePixelIndex = (activePixelIndex < numActivePixels);

	if (validActivePixelIndex) { // TODO: Or just return.

		// TODO: Restructure this block! It's a mess. Use classes!

		int pixelIndex = activePixels[activePixelIndex];

		thrust::default_random_engine rng( hash(seedOrPass) * hash(pixelIndex) * hash(rayDepth) );
		thrust::uniform_real_distribution<float> uniformDistribution(0,1);

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

			// TEST:
			Material bestMaterial;
			if (bestIsGroundPlane) {
				Material hardCodedGroundMaterial; // = makeEmptyMaterial();
				hardCodedGroundMaterial.diffuseColor = make_float3(0.455, 0.43, 0.39);
				hardCodedGroundMaterial.emittedColor = make_float3(0,0,0);
				hardCodedGroundMaterial.specularRefractiveIndex = 0;
				hardCodedGroundMaterial.hasTransmission = false;
				bestMaterial = hardCodedGroundMaterial;
			} else if (bestIsSphere) {
				bestMaterial = spheres[bestSphereIndex].material;
			}



			// TEST:
			// TODO: Reduce duplicate code and memory usage here and in the functions called here.
			// TODO: Finish implementing the functions called here.
			float3 incident = -rays[pixelIndex].direction;
			float3 reflectionDirection = computeReflectionDirection(bestNormal, incident);
			float3 transmissionDirection = computeTransmissionDirection(bestNormal, incident, AIR_IOR, bestMaterial.specularRefractiveIndex); // TODO: Detect total internal reflection!!!
			float3 biasVector = ( RAY_BIAS_DISTANCE * bestNormal ); // TODO: Bias ray in the other direction if the new ray is transmitted!!!
			float rouletteRandomFloat = uniformDistribution(rng);
			if (bestMaterial.specularRefractiveIndex > 1.0 && rouletteRandomFloat < computeFresnel(bestNormal, incident, AIR_IOR, bestMaterial.specularRefractiveIndex, reflectionDirection, transmissionDirection).reflectionCoefficient) {
				// Ray reflected from the surface. Trace a ray in the reflection direction.

				// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
				notAbsorbedColors[pixelIndex] *= bestMaterial.specularColor;

				rays[pixelIndex].origin = bestIntersectionPoint + biasVector;
				rays[pixelIndex].direction = reflectionDirection;
			} else {
				// Ray did not reflect from the surface, so consider emission and take a diffuse sample.

				// TODO: Use Russian roulette instead of simple multipliers! (Selecting between diffuse sample and no sample (absorption) in this case.)
				accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * bestMaterial.emittedColor;
				notAbsorbedColors[pixelIndex] *= bestMaterial.diffuseColor;

				// Choose a new ray direction:
				float randomFloat1 = uniformDistribution(rng); 
				float randomFloat2 = uniformDistribution(rng); 
				float3 newRayDirection = cosineWeightedDirectionInHemisphere(bestNormal, randomFloat1, randomFloat2);
				rays[pixelIndex].origin = bestIntersectionPoint + biasVector;
				rays[pixelIndex].direction = newRayDirection;
			}

		} else {
			// Ray didn't hit an object, so sample the background and terminate the ray.

			accumulatedColors[pixelIndex] += notAbsorbedColors[pixelIndex] * computeBackgroundColor(rays[pixelIndex].direction);
			notAbsorbedColors[pixelIndex] = make_float3(0,0,0); // The ray now has zero weight. TODO: Terminate the ray.
		}

		// To assist Thrust steram compaction, set this activePixel to -1 if the ray weight is now zero:
		// Brilliant (maybe).
		if (length(notAbsorbedColors[pixelIndex]) <= MIN_RAY_WEIGHT) { // Faster: dot product of a vector with itself is the same as its length squared.
			activePixels[activePixelIndex] = -1;
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
	Color* notAbsorbedColors = NULL;
	Color* accumulatedColors = NULL;
	CUDA_SAFE_CALL( cudaMalloc((void**)&activePixels, numPixels * sizeof(int)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&rays, numPixels * sizeof(Ray)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&notAbsorbedColors, numPixels * sizeof(Color)) );
	CUDA_SAFE_CALL( cudaMalloc((void**)&accumulatedColors, numPixels * sizeof(Color)) );

	initializeActivePixelsAndColors<<<fullBlocksPerGrid, threadsPerBlock>>>(numPixels, activePixels, notAbsorbedColors, accumulatedColors);

	int numActivePixels = numPixels;

	// Run this every pass so we can do anti-aliasing using jittering.
	// If we don't want to re-compute the camera rays, we'll need a separate array for secondary rays.
	
	raycastFromCameraKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(renderCam->position, renderCam->view, renderCam->up, renderCam->fov, renderCam->resolution, numPixels, rays, counter);





	for (int rayDepth = 0; rayDepth < MAX_TRACE_DEPTH; rayDepth++) {

		// Compute the number of blocks required, performing a ceiling operation to make sure there are enough:
		int newBlocksPerGrid = (numActivePixels + threadsPerBlock - 1) / threadsPerBlock; // Duplicate code.

		traceRayKernel<<<newBlocksPerGrid, threadsPerBlock>>>(numSpheres, spheres, numActivePixels, activePixels, rays, rayDepth, notAbsorbedColors, accumulatedColors, counter);
		


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
	CUDA_SAFE_CALL( cudaFree( notAbsorbedColors ) );
	CUDA_SAFE_CALL( cudaFree( accumulatedColors ) );


}
