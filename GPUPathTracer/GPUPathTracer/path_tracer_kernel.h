#ifndef PATH_TRACER_KERNEL_H
#define PATH_TRACER_KERNEL_H


// Necessary forward declaration:
extern "C"
void launch_kernel(int numSpheres, Sphere* spheres, int numPixels, Color* pixels, Ray* rays, int counter, RenderCamera* rendercam);


#endif // PATH_TRACER_KERNEL_H