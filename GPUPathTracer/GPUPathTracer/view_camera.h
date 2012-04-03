#ifndef VIEW_CAMERA_H
#define VIEW_CAMERA_H

#include "glm/glm.hpp"

class ViewCamera
{
public:
	ViewCamera();
	virtual ~ViewCamera();
   	void orbitLeft(float m);
	void orbitRight(float m);
	void orbitUp(float m);
	void orbitDown(float m);
	void zoomIn(float m);
	void zoomOut(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCam(Camera* rendercam);

	glm::vec4 up;
	glm::vec4 view;
	glm::vec4 eye;
	glm::mat4 rotation;
	glm::vec2 resolution;
	glm::vec2 fov;
};

#endif // VIEW_CAMERA_H