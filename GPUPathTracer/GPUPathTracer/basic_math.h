// Written by Peter Kutz.
// Used for math operations and constants that aren't included in the C++ standard library.
// Stripped down and modified version for GPU path tracer project.

#pragma once

#include "cutil_math.h"
#include "color.h" // Currently just for swap!

namespace BasicMath {

    extern const float PI;
    extern const float ONE_OVER_PI;
    extern const float TWO_PI;
    extern const float FOUR_PI;
    extern const float ONE_OVER_FOUR_PI;
	extern const float PI_OVER_TWO;
    extern const float E;
	extern const float SQRT_OF_ONE_THIRD;

    extern unsigned int mod(int x, int y); // Proper int mod with an always-positive result (unlike %).

    extern float mod(float x, float y); // Proper float mod with an always-positive result (unlike fmod).

    extern float radiansToDegrees(float radians);

    extern float degreesToRadians(float degrees);

    extern float average(float n1, float n2);
	
    extern float round(float n);

    extern float square(float n);

    extern float log2(float n); // std::log2 is not avaiable in all compilers.

    extern bool isNaN(float n); // std::isnan is not avaiable in all compilers.
	
	extern float min(float a, float b);

	extern float max(float a, float b);

    extern float clamp(float n, float low, float high);
	
    extern float repeat(float n, float modulus);

    extern int sign(float n);

    extern int positiveCharacteristic(float n);
	
    extern void swap(float & a, float & b); // Should this be here? TODO: Maybe make a swap template class instead!

}

