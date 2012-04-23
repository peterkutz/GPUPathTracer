#ifndef MEDIUM_H
#define MEDIUM_H

#include <cuda_runtime.h>
#include "absorption_and_scattering_properties.h"


struct Medium {

	float refractiveIndex;
	AbsorptionAndScatteringProperties absorptionAndScatteringProperties;

};


// TODO: Figure out a way to do this without a macro! Ideally, figure out how to use classes in CUDA.
#define AIR_IOR 1.000293 // Don't put this here!
#define SET_TO_AIR_MEDIUM(medium)																	 \
{									                                                                 \
	Medium airMedium;		                                                                         \
	airMedium.refractiveIndex = AIR_IOR;															 \
	SET_TO_AIR_ABSORPTION_AND_SCATTERING_PROPERTIES(airMedium.absorptionAndScatteringProperties);	 \
	medium = airMedium;																				 \
}

#endif // MEDIUM_H