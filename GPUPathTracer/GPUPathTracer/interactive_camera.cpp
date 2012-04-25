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
#include "interactive_camera.h"
#include "camera.h"

InteractiveCamera::InteractiveCamera() 
{   
	centerPosition = glm::vec3(0,0,0);
	yaw = 0;
	pitch = 0.3;
	radius = 4;
	apertureRadius = 0.04;

	resolution = glm::vec2(512,512);
	fov = glm::vec2(40, 40);
}

InteractiveCamera::~InteractiveCamera() {}

void InteractiveCamera::changeYaw(float m){
	yaw += m;
	fixYaw();
}

void InteractiveCamera::changePitch(float m){
	pitch += m;
	fixPitch();
}

void InteractiveCamera::changeRadius(float m){
	radius += radius * m; // Change proportional to current radius. Assuming radius isn't allowed to go to zero.
	fixRadius();
}

void InteractiveCamera::changeAltitude(float m){
	centerPosition.y += m;
	//fixCenterPosition();
}

void InteractiveCamera::changeApertureDiameter(float m){
	apertureRadius += (apertureRadius + 0.01) * m; // Change proportional to current apertureRadius.
	fixApertureRadius();
}

/*
void InteractiveCamera::changeFocalDistance(float m){
	focalDistance += m;
	fixFocalDistance();
}
*/

void InteractiveCamera::setResolution(float x, float y){
	resolution = glm::vec2(x,y);
	setFOVX(fov.x);
}

void InteractiveCamera::setFOVX(float fovx){
	fov.x = fovx;
	fov.y = BasicMath::radiansToDegrees( atan( tan(BasicMath::degreesToRadians(fovx) * 0.5) * (resolution.y / resolution.x) ) * 2.0 );
	//fov.y = (fov.x*resolution.y)/resolution.x; // TODO: Fix this! It's not correct! Need to use trig!
}

void InteractiveCamera::buildRenderCamera(Camera* renderCamera){
	float xDirection = sin(yaw) * cos(pitch);
	float yDirection = sin(pitch);
	float zDirection = cos(yaw) * cos(pitch);
	glm::vec3 directionToCamera = glm::vec3(xDirection, yDirection, zDirection);
	glm::vec3 viewDirection = -directionToCamera;
	glm::vec3 eyePosition = centerPosition + directionToCamera * radius;

	renderCamera->position = make_float3(eyePosition[0], eyePosition[1], eyePosition[2]);
	renderCamera->view = make_float3(viewDirection[0], viewDirection[1], viewDirection[2]);
	renderCamera->up = make_float3(0, 1, 0);
	renderCamera->resolution = make_float2(resolution.x, resolution.y);
	renderCamera->fov = make_float2(fov.x, fov.y);
	renderCamera->apertureRadius = apertureRadius;
	renderCamera->focalDistance = radius;
}

void InteractiveCamera::fixYaw() {
	yaw = BasicMath::mod(yaw, BasicMath::TWO_PI); // Normalize the yaw.
}

void InteractiveCamera::fixPitch() {
	float padding = 0.05;
	pitch = BasicMath::clamp(pitch, -BasicMath::PI_OVER_TWO + padding, BasicMath::PI_OVER_TWO - padding); // Limit the pitch.
}

void InteractiveCamera::fixRadius() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = BasicMath::clamp(radius, minRadius, maxRadius);
}

void InteractiveCamera::fixApertureRadius() {
	float minApertureRadius = 0.0;
	float maxApertureRadius = 25.0;
	apertureRadius = BasicMath::clamp(apertureRadius, minApertureRadius, maxApertureRadius);
}

/*
void InteractiveCamera::fixFocalDistance() {
	float minRadius = 0.2;
	float maxRadius = 100.0;
	radius = BasicMath::clamp(radius, minRadius, maxRadius);
}
*/