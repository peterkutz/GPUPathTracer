#ifndef RENDERCAMERA_H
#define RENDERCAMERA_H

#include <cuda_runtime.h>

struct RenderCamera {
	float2 resolution;
	float3 position;
	float3 view;
	float3 up;
	float2 fov; // TODO: Define one angle, derive the other one.
};

#endif // RENDERCAMERA_H