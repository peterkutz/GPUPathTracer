#ifndef SPHERE_H
#define SPHERE_H

#include <cuda_runtime.h>
#include "color.h"

struct Sphere {

	float3 position;
	float radius;
	
	Color diffuseColor; //Material* material;
	Color emittedColor;

};

#endif // SPHERE_H