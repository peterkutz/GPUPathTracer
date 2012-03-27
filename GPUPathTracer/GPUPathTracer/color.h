#ifndef COLOR_H
#define COLOR_H


#include <cuda_runtime.h>
#include "cutil_math.h"


typedef float3 Color;


float& component(Color & color, int componentIndex);
const float& readOnlyComponent(const Color & color, int componentIndex);
Color gammaCorrect(const Color & color);
uchar3 floatTo8Bit(const Color & color);


#endif // COLOR_H