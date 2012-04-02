#ifndef SPHERE_H
#define SPHERE_H

#include <cuda_runtime.h>
#include "material.h"

struct Sphere {

	float3 position;
	float radius;
	
	Material material;

};

#endif // SPHERE_H