#ifndef INTERACTIVE_CAMERA_H
#define INTERACTIVE_CAMERA_H

#include "glm/glm.hpp"

class InteractiveCamera
{
private:

	glm::vec3 centerPosition;
	float yaw;
	float pitch;
	float radius;
	float apertureRadius;

	void fixYaw();
	void fixPitch();
	void fixRadius();
	void fixApertureRadius();

public:
	InteractiveCamera();
	virtual ~InteractiveCamera();
   	void changeYaw(float m);
	void changePitch(float m);
	void changeRadius(float m);
	void changeAltitude(float m);
	void changeApertureDiameter(float m);
	void setResolution(float x, float y);
	void setFOVX(float fovx);

	void buildRenderCamera(Camera* renderCamera);

	glm::vec2 resolution;
	glm::vec2 fov;
};

#endif // INTERACTIVE_CAMERA_H