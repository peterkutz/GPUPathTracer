#include "color.h"
#include "basic_math.h"
#include <cmath>
#include <cassert>


float& component(Color & color, int componentIndex) {

	assert(componentIndex == 0 || componentIndex == 1 || componentIndex == 2);

	switch(componentIndex) {
		case 0:
			return color.x;
		case 1:
			return color.y;
		case 2:
			return color.z;
		default:
			return color.x;
	}

}

const float& readOnlyComponent(const Color & color, int componentIndex) {

	assert(componentIndex == 0 || componentIndex == 1 || componentIndex == 2);

	switch(componentIndex) {
		case 0:
			return color.x;
		case 1:
			return color.y;
		case 2:
			return color.z;
		default:
			return color.x;
	}

}

Color gammaCorrect(const Color & color) {
	// Simple power law gamma for now.
	// TODO: Use sRGB.
	float gamma = 2.2;
	float oneOverGamma = 1.0 / gamma;
	Color gammaCorrectedColor;
	for (int i = 0; i < 3; i++) {
		component(gammaCorrectedColor, i) = std::pow(readOnlyComponent(color, i), oneOverGamma);
	}
	return gammaCorrectedColor;
}

uchar3 floatTo8Bit(const Color & color) {
	// TODO: Perform dithering in this function, too!

	Color gammaCorrectedColor = gammaCorrect(color);
	uchar3 eightBitColor;
	eightBitColor.x = BasicMath::clamp(gammaCorrectedColor.x * 255, 0, 255);
	eightBitColor.y = BasicMath::clamp(gammaCorrectedColor.y * 255, 0, 255);
	eightBitColor.z = BasicMath::clamp(gammaCorrectedColor.z * 255, 0, 255);
	return eightBitColor;
}

