#pragma once

// TODO: Which of these are necessary here?
#include <cuda_runtime.h>

class PathTracer {

private:

	int imageWidth;
	int imageHeight;
	int numPixels;

	float3* image;

	int numSpheres;

	float3* spherePositions;
	float3* sphereRadii;
	float3* sphereDiffuseColors;
	float3* sphereEmittedColors;
	
	float3* rayOrigins;
	float3* rayDirections;

	float3* currentImage;

	void createDeviceData();
	void deleteDeviceData();

	void runCuda();

	void setUpScene();

public:
	PathTracer();
	~PathTracer();

	void setImageSize(int _imageWidth, int _imageHeight);
	float3* render();


};