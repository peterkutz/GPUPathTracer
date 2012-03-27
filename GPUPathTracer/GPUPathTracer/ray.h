#ifndef RAY_H
#define RAY_H

#include <cuda_runtime.h>

struct Ray {

	float3 origin;
	float3 direction;

};

#endif // RAY_H