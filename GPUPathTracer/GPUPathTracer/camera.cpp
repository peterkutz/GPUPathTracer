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

glm::vec3  Camera::dfltEye(5.0, 1.0, -25.0);
glm::vec3  Camera::dfltUp(0.0, 1.0, 0.0);
glm::vec3  Camera::dfltLook(0.5, 0.5, 0.5);
float Camera::dfltVfov = 60.0;
float Camera::dfltAspect = 1.0;
float Camera::dfltNear = 0.5;
float Camera::dfltFar = 120.0;
float Camera::dfltSpeed = 0.1;
float Camera::dfltTurnRate = 1.0*(BasicMath::PI/180.0);

Camera::Camera() 
{   
   myDir = NONE; myTurnDir = NONE;
   reset();
}

Camera::~Camera() {}

void Camera::reset()
{
   mSpeed = dfltSpeed;
   mTurnRate = dfltTurnRate;
   mVfov = dfltVfov;
   mAspect = dfltAspect;
   mNear = dfltNear;
   mFar = dfltFar;

   // Calculate the initial heading & pitch
   // Note that  eye[0] = radius*cos(h)*cos(p); and  eye[1] = radius*sin(p);
   mPitch = -std::asin(dfltEye[1]/dfltEye.length());
   mHeading = std::acos(dfltEye[0]/(dfltEye.length()*std::cos(mPitch)));
   //printf("INIT: %f %f\n", mPitch, mHeading);

   set(dfltEye, dfltLook, dfltUp);
}


void Camera::draw()
{
   glMatrixMode(GL_PROJECTION);
   glLoadIdentity();
   gluPerspective(mVfov, mAspect, mNear, mFar);


   float m[16];
  /* m[0] = v[0]; m[4] = v[1]; m[8] = v[2];  m[12] = -glm::dot(eye, v); 
   m[1] = u[0]; m[5] = u[1]; m[9] = u[2];  m[13] = -glm::dot(eye, u); 
   m[2] = n[0]; m[6] = n[1]; m[10] = n[2]; m[14] = -glm::dot(eye, n); 
   m[3] = 0.0;  m[7] = 0.0;  m[11] = 0.0;  m[15] = 1.0;*/



   m[0] = lookMat[0][0]; m[4] = lookMat[1][0]; m[8]  = lookMat[2][0];  m[12] = lookMat[3][0];
   m[1] = lookMat[0][1]; m[5] = lookMat[1][1]; m[9]  = lookMat[2][1];  m[13] = lookMat[3][1];
   m[2] = lookMat[0][2]; m[6] = lookMat[1][2]; m[10] = lookMat[2][2];  m[14] = lookMat[3][2];
   m[3] = lookMat[0][3]; m[7] = lookMat[1][3]; m[11] = lookMat[2][3];  m[15] = lookMat[3][3];


   glMatrixMode(GL_MODELVIEW);
   glLoadMatrixf(m); 

   glGetDoublev(GL_MODELVIEW_MATRIX, myModelMatrix);
   glGetDoublev(GL_PROJECTION_MATRIX, myProjMatrix);
   glGetIntegerv(GL_VIEWPORT, myViewport);
} 

const glm::vec3& Camera::getUp() const
{
   return u;
}

const glm::vec3& Camera::getBackward() const
{
   return n;
}

const glm::vec3& Camera::getRight() const
{
   return v;
}

glm::vec3 Camera::getRelativePosition(float left, float up, float forward)
{
   glm::vec3 direction = up*u + left*v - forward*n;
   return eye + direction;  // Move along forward axis 
}

void Camera::getViewport(int& x, int& y, int& w, int& h)
{
   x = myViewport[0];
   y = myViewport[1];
   w = myViewport[2];
   h = myViewport[3];
}

void Camera::getProjection(
   float* vfov, float* aspect, float* zNear, float* zFar)
{
   *vfov = mVfov; *aspect = mAspect; *zNear = mNear; *zFar = mFar;
}

void Camera::setPosition(const glm::vec3& pos)
{
   eye = pos;
}

const glm::vec3& Camera::getPosition() const
{
   return eye;
}

void Camera::buildRenderCam(RenderCamera* rendercam){
	
	float3 rendercam_eye = make_float3(eye[0], eye[1], eye[2]);
	float3 rendercam_view = make_float3(-n[0], -n[1], -n[2]);
	float3 rendercam_up = make_float3(u[0], u[1], u[2]);

	rendercam->position = rendercam_eye;
	rendercam->view = rendercam_view;
	rendercam->up = rendercam_up;
}

void Camera::setProjection(
   float vfov, float aspect, float zNear, float zFar)
{
   mVfov = vfov;
   mAspect = aspect;
   mNear = zNear;
   mFar = zFar;
}

float Camera::heading() const
{
   return mHeading;
}

float Camera::pitch() const
{
   return mPitch;
}

void Camera::set(const glm::vec3& eyepos, const glm::vec3& look, const glm::vec3& up)
{
	eye = eyepos;
	n = eyepos - look;
	v = glm::cross(up, n);
	u = glm::cross(n, v);
	mRadius = n.length()/2; // cache this distance
	lookMat = glm::gtx::transform2::lookAt(eyepos, look, up);
	u = glm::normalize(u);
	v = glm::normalize(v);
	n = glm::normalize(n);
}

void Camera::move(float dV, float dU, float dN)
{
   eye += dU*u + dV*v + dN*n;
}

void Camera::orbit(float h, float p)
{
  //printf("PITCH: %f\n", p);
  //printf("HEADING: %f\n", h);
  //printf("RADIUS: %f\n", mRadius);

   glm::vec3 rotatePt; // Calculate new location around sphere having mRadius
   rotatePt[0] = mRadius*std::cos(h)*std::cos(p);
   rotatePt[1] = mRadius*std::sin(p);
   rotatePt[2] = mRadius*std::sin(h)*std::cos(p);

   glm::vec3 lookAt = eye-n*mRadius;


   set(lookAt-rotatePt, lookAt /* look */, glm::vec3(0.0f, 1.0f, 0.0f) /* up Approx */);
}
  
void Camera::orbitLeft(int scale) 
{
   myTurnDir = TL;
   mHeading -= mTurnRate*scale;
   orbit(mHeading, pitch());
}

void Camera::moveLeft(int scale) // => move along v
{    
   myDir = L;
   move(-mSpeed*scale, 0.0, 0.0);
}

void Camera::orbitRight(int scale)
{
   myTurnDir = TR;
   mHeading += mTurnRate*scale;
   orbit(mHeading, pitch());
}

void Camera::moveRight(int scale) // => move along v
{
   myDir = R;
   move(mSpeed*scale, 0.0, 0.0);   
}

void Camera::orbitUp(int scale)
{
   myTurnDir = TU; 
   mPitch = glm::min(BasicMath::PI/2.0-0.01, double(mPitch + mTurnRate*scale));
   orbit(heading(), mPitch);
}

void Camera::moveUp(int scale) // => move along +u
{
   myDir = U;
   move(0.0, mSpeed*scale, 0.0);   
}

void Camera::orbitDown(int scale)
{
   myTurnDir = TD; 
   double d = -BasicMath::PI/2.0+0.01;
   double d2 = mPitch - mTurnRate*scale;
   mPitch = glm::max(d, d2);
   orbit(heading(), mPitch);
}

void Camera::moveDown(int scale) // => move along -u
{
   myDir = D;
   move(0.0, -mSpeed*scale, 0.0);   
}

void Camera::moveForward(int scale) // => move along -n
{
   myDir = F; 
   move(0.0, 0.0, -mSpeed*scale);      
   mRadius += -mSpeed*scale;  // Also "zoom" into radius
}

void Camera::moveBack(int scale) // => move along n
{
   myDir = B; 
   move(0.0, 0.0, mSpeed*scale);   
   mRadius += mSpeed*scale;  // Also "zoom" out radius
}

void Camera::turn(glm::vec3& v1, glm::vec3& v2, float amount)
{
   double cosTheta = std::cos(amount);
   double sinTheta = std::sin(amount);

   float vX =  cosTheta*v1[0] + sinTheta*v2[0]; 
   float vY =  cosTheta*v1[1] + sinTheta*v2[1]; 
   float vZ =  cosTheta*v1[2] + sinTheta*v2[2]; 

   float nX = -sinTheta*v1[0] + cosTheta*v2[0]; 
   float nY = -sinTheta*v1[1] + cosTheta*v2[1]; 
   float nZ = -sinTheta*v1[2] + cosTheta*v2[2]; 

   v1[0] = vX; v1[1] = vY; v1[2] = vZ;
   v2[0] = nX; v2[1] = nY; v2[2] = nZ;
}

void Camera::turnLeft(int scale) // rotate around u
{
   myTurnDir = TL; 
   turn(v, n, -mTurnRate*scale);
}

void Camera::turnRight(int scale) // rotate around u
{
   myTurnDir = TR;
   turn(v, n, mTurnRate*scale);
}

void Camera::turnUp(int scale) // rotate around v
{
   myTurnDir = TU; 
   turn(n, u, mTurnRate*scale);
}

void Camera::turnDown(int scale) // rotate around v
{
   myTurnDir = TD; 
   turn(n, u, -mTurnRate*scale);
}

bool Camera::screenToWorld(int screenX, int screenY, glm::vec3& worldCoords)
{
   double x, y, z;
   GLint result = gluUnProject(screenX, screenY, 0.0, 
                               myModelMatrix, myProjMatrix, myViewport, 
                               &x, &y, &z);
   
   worldCoords[0] = x; worldCoords[0] = y; worldCoords[0] = z; 
   return result == GL_TRUE;
}

bool Camera::worldToScreen(const glm::vec3& worldCoords, int& screenX, int& screenY)
{
   double x, y, z;
   GLint result = gluProject(worldCoords[0], worldCoords[1], worldCoords[2],
                             myModelMatrix, myProjMatrix, myViewport, 
                             &x, &y, &z);

   screenX = (int) x;
   screenY = (int) y;
   return result == GL_TRUE;
}

glm::mat4 Camera::cameraToWorldMatrix()
{
   glm::mat4 tmp(glm::vec4(myModelMatrix[0], myModelMatrix[1], myModelMatrix[2], myModelMatrix[3]),
   glm::vec4(myModelMatrix[4], myModelMatrix[5], myModelMatrix[6], myModelMatrix[7]),
   glm::vec4(myModelMatrix[8], myModelMatrix[9], myModelMatrix[10], myModelMatrix[11]),
   glm::vec4(myModelMatrix[12], myModelMatrix[13], myModelMatrix[14], myModelMatrix[15]));
   tmp = glm::inverse(tmp);
   return tmp;
}
