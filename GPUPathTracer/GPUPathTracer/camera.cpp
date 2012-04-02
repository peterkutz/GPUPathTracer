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
#include "rendercamera.h"

Camera::Camera() 
{   
   up = glm::vec4(0,1,0,0);
   eye = glm::vec4(0,0,4,0);
   view = glm::vec4(0,0,0,0);
}

Camera::~Camera() {}

void Camera::orbitLeft(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),-m,glm::vec3(up));
	up = rotMat*up;
	eye= rotMat*eye;
}

void Camera::orbitRight(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),m,glm::vec3(up));
	up = rotMat*up;
	eye= rotMat*eye;
}

void Camera::orbitUp(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),-m,glm::cross(glm::vec3(eye),glm::vec3(up)));
	up = rotMat*up;
	eye= rotMat*eye;
}

void Camera::orbitDown(float m){
	glm::mat4 rotMat = glm::rotate(glm::mat4(),m,glm::cross(glm::vec3(eye),glm::vec3(up)));
	up = rotMat*up;
	eye= rotMat*eye;
}

void Camera::zoomIn(float m){
	glm::vec3 viewVec = m*glm::vec3(glm::normalize(view-eye));
	eye = eye + glm::vec4(viewVec,0);
}

void Camera::zoomOut(float m){
	glm::vec3 viewVec = -m*glm::vec3(glm::normalize(view-eye));
	eye = eye + glm::vec4(viewVec,0);
}


void Camera::buildRenderCam(RenderCamera* rendercam){
	glm::vec3 viewVec = glm::vec3(glm::normalize(view-eye));

	rendercam->position = make_float3(eye[0], eye[1], eye[2]);
	rendercam->view = make_float3(viewVec[0], viewVec[1], viewVec[2]);
	rendercam->up = make_float3(up[0], up[1], up[2]);
}