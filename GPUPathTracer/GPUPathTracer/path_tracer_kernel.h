#ifndef PATH_TRACER_KERNEL_H
#define PATH_TRACER_KERNEL_H


// Necessary forward declaration:
extern "C"
void launchKernel(int numSpheres, Sphere* spheres, int numPixels, Color* pixels, int counter, Camera* renderCam);


#endif // PATH_TRACER_KERNEL_H