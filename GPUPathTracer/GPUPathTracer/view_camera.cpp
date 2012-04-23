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
	centerPosition = glm::vec3(0,0,0);
	yaw = 0;
	pitch = 0.3;
	radius = 4;
	resolution = glm::vec2(512,512);
	fov = glm::vec2(40, 40);
}

ViewCamera::~ViewCamera() {}

void ViewCamera::changeYaw(float m){
	yaw += m;
	fixYaw();
}

void ViewCamera::changePitch(float m){
	pitch += m;
	fixPitch();
}

void ViewCamera::changeRadius(float m){
	radius += radius * m;
	fixRadius();
}

void ViewCamera::changeAltitude(float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void ViewCamera::setResolution(float x, float y){
	resolution = glm::vec2(x,y);
	setFOVX(fov.x);
}

void ViewCamera::setFOVX(float fovx){
	fov.x = fovx;
	fov.y = (fov.x*resolution.y)/resolution.x; // TODO: Fix this! It's not correct.
}

void ViewCamera::buildRenderCam(Camera* renderCam){
	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	glm::vec3 directionToCamera = glm::vec3(xDirection, yDirection, zDirection);
	glm::vec3 viewDirection = -directionToCamera;
	glm::vec3 eyePosition = centerPosition + directionToCamera * radius;

	renderCam->position = make_float3(eyePosition[0], eyePosition[1], eyePosition[2]);
	renderCam->view = make_float3(viewDirection[0], viewDirection[1], viewDirection[2]);
	renderCam->up = make_float3(0, 1, 0);
	renderCam->resolution = make_float2(resolution.x, resolution.y);
	renderCam->fov = make_float2(fov.x, fov.y);
}

void ViewCamera::fixYaw() {
	yaw = BasicMath::mod(yaw, BasicMath::TWO_PI); // Normalize the yaw.
}

void ViewCamera::fixPitch() {
	float padding = 0.05;
	pitch = BasicMath::clamp(pitch, -BasicMath::PI_OVER_TWO + padding, BasicMath::PI_OVER_TWO - padding); // Limit the pitch.
}

void ViewCamera::fixRadius() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = BasicMath::clamp(radius, minRadius, maxRadius);
}