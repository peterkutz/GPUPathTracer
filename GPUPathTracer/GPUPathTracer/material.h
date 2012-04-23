#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda_runtime.h>
#include "color.h"
#include "medium.h"

struct Material {

	Color diffuseColor;
	Color emittedColor;
	Color specularColor;
	bool hasTransmission;
	Medium medium;

};

// TODO: Figure out a way to do this without a macro! Ideally, figure out how to use classes in CUDA.
#define SET_DEFAULT_MATERIAL_PROPERTIES(material)															\
{																											\
	material.diffuseColor = make_float3(0,0,0);																\
	material.emittedColor = make_float3(0,0,0);																\
	material.specularColor = make_float3(0,0,0);															\
	material.hasTransmission = false;																		\
	SET_TO_AIR_MEDIUM(material.medium);																	\
}

/*
__host__ __device__
Material makeEmptyMaterial();
*/

#endif // MATERIAL_H