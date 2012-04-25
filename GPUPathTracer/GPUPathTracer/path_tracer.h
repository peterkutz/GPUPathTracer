#ifndef PATH_TRACER_H
#define PATH_TRACER_H

#include <cuda_runtime.h>
struct Image;
struct Sphere;
struct Ray;
struct Camera;
struct Poly;

class PathTracer {

private:

	Image* image;

	int numSpheres;
	int numPolys;
	Sphere* spheres;
	Poly* polys;
	Poly* dev_polys;
	void createDeviceData();
	void deleteDeviceData();

	void setUpScene();

public:
	PathTracer(Camera* cam);
	~PathTracer();

	void reset();

	Image* render();

	Camera* renderCamera;

	void prepMesh();
};

#endif // PATH_TRACER_H