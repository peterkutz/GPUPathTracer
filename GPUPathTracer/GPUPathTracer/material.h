#ifndef MATERIAL_H
#define MATERIAL_H

#include <cuda_runtime.h>
#include "color.h"

struct Material {

	Color diffuseColor;
	Color emittedColor;
	Color specularColor;
	float specularRefractiveIndex;
	bool hasTransmission;

};

Material makeEmptyMaterial();

#endif // MATERIAL_H