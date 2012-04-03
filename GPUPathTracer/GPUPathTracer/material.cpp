#include "material.h"

Material makeEmptyMaterial() {
	Material material;
	material.diffuseColor = make_float3(0,0,0);
	material.emittedColor = make_float3(0,0,0);
	material.specularColor = make_float3(0,0,0);
	material.specularRefractiveIndex = 0;
	material.hasTransmission = false;
	return material;
}


