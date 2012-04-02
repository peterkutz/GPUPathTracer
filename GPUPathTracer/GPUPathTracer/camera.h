
#ifndef camera_H_
#define camera_H_

#include "glm/glm.hpp"

struct RenderCamera;

class Camera
{
public:
	Camera();
	virtual ~Camera();
   	void orbitLeft(float m);
	void orbitRight(float m);
	void orbitUp(float m);
	void orbitDown(float m);
	void zoomIn(float m);
	void zoomOut(float m);

	void buildRenderCam(RenderCamera* rendercam);

	glm::vec4 up;
	glm::vec4 view;
	glm::vec4 eye;
	glm::mat4 rotation;
};

#endif
