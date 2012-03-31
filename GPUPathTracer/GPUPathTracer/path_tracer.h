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

	void setUpScene();

	int counter;

public:
	PathTracer();
	~PathTracer();

	Image* render();


};

#endif // PATH_TRACER_H