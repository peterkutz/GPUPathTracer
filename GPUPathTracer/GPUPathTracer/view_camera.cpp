#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cmath>
#include <time.h>
#include "basic_math.h"
#include <iostream>
#include "camera.h"
#include "glm/gtx/transform2.hpp"
#include "view_camera.h"
#include "camera.h"

ViewCamera::ViewCamera() 
{   
   up = glm::vec4(0,1,0,0);
   eye = glm::vec4(0,0,4,0);
   view = glm::vec4(0,0,0,0);
   resolution = glm::vec2(512,512);
   fov = glm::vec2(40, 40);
}

ViewCamera::~ViewCamera() {}

void ViewCamera::orbitLeft(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),-m,glm::vec3(up));
	up = rotMat*up;
	eye= rotMat*eye;
}

void ViewCamera::orbitRight(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),m,glm::vec3(up));
	up = rotMat*up;
	eye= rotMat*eye;
}

void ViewCamera::orbitUp(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),-m,glm::cross(glm::vec3(eye),glm::vec3(up)));
	up = rotMat*up;
	eye= rotMat*eye;
}

void ViewCamera::orbitDown(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),m,glm::cross(glm::vec3(eye),glm::vec3(up)));
	up = rotMat*up;
	eye= rotMat*eye;
}

void ViewCamera::zoomIn(float m){
	glm::vec3 viewVec = m*glm::vec3(glm::normalize(view-eye));
	eye = eye + glm::vec4(viewVec,0);
}

void ViewCamera::zoomOut(float m){
	glm::vec3 viewVec = -m*glm::vec3(glm::normalize(view-eye));
	eye = eye + glm::vec4(viewVec,0);
}

void ViewCamera::setResolution(float x, float y){
	resolution = glm::vec2(x,y);
	setFOVX(fov.x);
}

void ViewCamera::setFOVX(float fovx){
	fov.x = fovx;
	fov.y = (fov.x*resolution.y)/resolution.x;
}

void ViewCamera::buildRenderCam(Camera* rendercam){
	glm::vec3 viewVec = glm::vec3(glm::normalize(view-eye));

	rendercam->position = make_float3(eye[0], eye[1], eye[2]);
	rendercam->view = make_float3(viewVec[0], viewVec[1], viewVec[2]);
	rendercam->up = make_float3(up[0], up[1], up[2]);
	rendercam->resolution = make_float2(resolution.x, resolution.y);
	rendercam->fov = make_float2(fov.x, fov.y);
}

