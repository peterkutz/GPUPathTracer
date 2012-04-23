#ifndef UTILITY_H_
#define UTILITY_H_

#include <GL/glew.h>

namespace Utility
{

GLuint createProgram(const char *vertexShaderPath, const char *fragmentShaderPath, const char *attributeLocations[], GLuint numberOfLocations);

}
 
#endif