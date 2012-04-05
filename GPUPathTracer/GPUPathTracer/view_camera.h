#ifndef VIEW_CAMERA_H
#define VIEW_CAMERA_H

#include "glm/glm.hpp"

class ViewCamera
{
private:

	glm::vec3 centerPosition;
	float yaw;
	float pitch;
	float radius;

	void fixYaw();
	void fixPitch();
	void fixRadius();

public:
	ViewCamera();
	virtual ~ViewCamera();
   	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCam(Camera* renderCam);

	glm::vec2 resolution;
	glm::vec2 fov;
};

#endif // VIEW_CAMERA_H