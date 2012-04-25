#ifndef PATH_TRACER_KERNEL_H
#define PATH_TRACER_KERNEL_H

/*#include "cukd/kdtree.h"
#include "cukd/primitives.h"
#include "cukd/utils.h"*/

// Necessary forward declaration:
extern "C"
void launchKernel(int numPolys, Poly* polys, int numSpheres, Sphere* spheres, int numPixels, Color* pixels, int counter, Camera renderCamera);
/*extern "C"
void constructKDTree(TriangleArray triarr, float minx, float miny, float minz, float maxx, float maxy, float maxz);
*/
#endif // PATH_TRACER_KERNEL_H