#ifndef CAMERA_H
#define CAMERA_H

#include <cuda_runtime.h>

struct Camera {
	float2 resolution;
	float3 position;
	float3 view;
	float3 up;
	float2 fov; 
	float apertureRadius;
	float focalDistance;
};

#endif // CAMERA_H