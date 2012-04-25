#ifndef POLY_H
#define POLY_H

#include <cuda_runtime.h>
#include "material.h"

struct Poly {

	float3 position;
	float3 p0;
	float3 p1;
	float3 p2;
	float3 n;
	
	Material material;

};

#endif // SPHERE_H