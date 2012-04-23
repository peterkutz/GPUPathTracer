#ifndef ABSORPTION_AND_SCATTERING_PROPERTIES_H
#define ABSORPTION_AND_SCATTERING_PROPERTIES_H

#include <cuda_runtime.h>
#include "color.h"

struct AbsorptionAndScatteringProperties {

	Color absorptionCoefficient;
	float reducedScatteringCoefficient;

};

// TODO: Figure out a way to do this without a macro! Ideally, figure out how to use classes in CUDA.
#define SET_TO_AIR_ABSORPTION_AND_SCATTERING_PROPERTIES(absorptionAndScatteringProperties)		            \
{																						                        \
	AbsorptionAndScatteringProperties airAbsorptionAndScatteringProperties = {make_float3(0,0,0), 0};		    \
	absorptionAndScatteringProperties = airAbsorptionAndScatteringProperties;									\
}

#endif // ABSORPTION_AND_SCATTERING_PROPERTIES_H