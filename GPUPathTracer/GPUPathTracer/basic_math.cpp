// Written by Peter Kutz.

#include "basic_math.h"
#include <cmath>

const float BasicMath::PI =					3.1415926535897932384626422832795028841971;
const float BasicMath::ONE_OVER_PI =		0.3183098861837906715377675267450287240689;
const float BasicMath::TWO_PI =				6.2831853071795864769252867665590057683943;
const float BasicMath::FOUR_PI =			12.566370614359172953850573533118011536788;
const float BasicMath::ONE_OVER_FOUR_PI =	0.0795774715459476678844418816862571810172;
const float BasicMath::PI_OVER_TWO =		1.5707963267948966192313216916397514420985;
const float BasicMath::E =                  2.7182818284590452353602874713526624977572;
const float BasicMath::SQRT_OF_ONE_THIRD =  0.5773502691896257645091487805019574556476;

// TODO: Make these inline functions?

// Is the int version necessary???
unsigned int BasicMath::mod(int x, int y) { // Doesn't account for -y ???
	int result = x % y;
	if (result < 0) result += y;
	return result;
}

float BasicMath::mod(float x, float y) { // Does this account for -y ???
	return x - y * std::floor(x / y);
}

float BasicMath::radiansToDegrees(float radians) {
	float degrees = radians * 180.0 / BasicMath::PI;
	return degrees;
}

float BasicMath::degreesToRadians(float degrees) {
	float radians = degrees / 180.0 * BasicMath::PI;
	return radians;
}

float BasicMath::average(float n1, float n2) {
	return ((n1 + n2) / 2);
}

float BasicMath::round(float n) {
	return std::floor(n + 0.5);
}

float BasicMath::square(float n) {
    return n * n;
}

float BasicMath::log2(float n) {
    return std::log(n) / std::log(2.0);
}

bool BasicMath::isNaN(float n) {
    return (n != n);
}

float BasicMath::min(float a, float b) {
	if (a < b) {
		return a;
	} else {
		return b;
	}
}

float BasicMath::max(float a, float b) {
	if (a > b) {
		return a;
	} else {
		return b;
	}
}

float BasicMath::clamp(float n, float low, float high) {
	n = min(n, high); // Was std::min.
	n = max(n, low); // Was std::max.
	return n;
}
float BasicMath::repeat(float n, float modulus) {
	// http://en.wikipedia.org/wiki/Modular_arithmetic
	n = BasicMath::mod(n, modulus);
	return n;
}

int BasicMath::sign(float n) {
    if (n >= 0) {
        return 1;
    } else {
        return -1;
    }
}
int BasicMath::positiveCharacteristic(float n) {
    if (n > 0) {
        return 1;
    } else {
        return 0;
    }
}

void BasicMath::swap(float & a, float & b) {
    float temp = a;
    a = b;
    b = temp;
}





