#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <cuda_runtime.h>
struct Image;
struct Sphere;
struct Ray;

class PathTracer {

private:

	Image* image;

	int numSpheres;

	Sphere* spheres;
	
	Ray* rays;

	void createDeviceData();
	void deleteDeviceData();

	void runCuda();

	void setUpScene();

public:
	PathTracer();
	~PathTracer();

	Image* render();


};

#endif // PATH_TRACER_H