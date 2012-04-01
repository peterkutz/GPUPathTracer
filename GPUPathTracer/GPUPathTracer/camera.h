
#ifndef camera_H_
#define camera_H_

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <windows.h>
#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>
#endif

#include "glm/glm.hpp"

struct RenderCamera;

class Camera
{
public:
   Camera();
   virtual ~Camera();

   // Draw projection and eyepoint
   virtual void draw();
   virtual void buildRenderCam(RenderCamera* rendercam);

   // Initialize the camera with glyLookAt parameters
   virtual void set(const glm::vec3& eyepos, const glm::vec3& look, const glm::vec3& up);

   // Get camera state
   virtual void setPosition(const glm::vec3& pos);
   virtual const glm::vec3& getPosition() const;
   virtual const glm::vec3& getUp() const;
   virtual const glm::vec3& getBackward() const;
   virtual const glm::vec3& getRight() const;
   virtual glm::vec3 getRelativePosition(float left, float up, float forward);
   virtual float heading() const;
   virtual float pitch() const;

   // Camera frustrum managements
   virtual void setProjection(
      float vfov, float aspect, float zNear, float zFar);
   virtual void getProjection(
      float* vfov, float* aspect, float* zNear, float* zFar);
   virtual void getViewport(int& x, int& y, int& w, int& h);

   // Relative movement commands
   virtual void moveLeft(int scale = 1.0);
   virtual void moveRight(int scale = 1.0);
   virtual void moveUp(int scale = 1.0);
   virtual void moveDown(int scale = 1.0);
   virtual void moveForward(int scale = 1.0);
   virtual void moveBack(int scale = 1.0);

   virtual void turnLeft(int scale = 1.0);
   virtual void turnRight(int scale = 1.0);
   virtual void turnUp(int scale = 1.0);
   virtual void turnDown(int scale = 1.0);

   virtual void orbitLeft(int scale = 1.0);
   virtual void orbitRight(int scale = 1.0);
   virtual void orbitUp(int scale = 1.0);
   virtual void orbitDown(int scale = 1.0);

   // Reset to original state
   virtual void reset();

   // Conversion utilities between screen and world coordinates
   virtual bool screenToWorld(int screenX, int screenY, glm::vec3& worldCoords);
   virtual bool worldToScreen(const glm::vec3& worldCoords, int& screenX, int& screenY);

   // Get camera to world matrix
   virtual glm::mat4 cameraToWorldMatrix();

protected:
   enum Dir { NONE, F, B, L, R, U, D, TL, TR, TU, TD} myDir, myTurnDir;
   virtual void turn(glm::vec3& v, glm::vec3& n, float amount);
   virtual void move(float dU, float dV, float dN);
   virtual void orbit(float h, float p);

protected:
   float mSpeed, mTurnRate;

   glm::vec3 eye; // camera position
   float mHeading, mPitch, mRadius;
   float mVfov, mAspect, mNear, mFar; // projection parameters
   

   // Basis of camera local coord system
   glm::vec3 u; // up
   glm::vec3 v; // v points right
   glm::vec3 n; // -n points forward

   // Cache useful values
   GLdouble myProjMatrix[16];
   GLdouble myModelMatrix[16];
   GLint myViewport[4];

public:

   // Defaults
   static glm::vec3 dfltEye, dfltUp, dfltLook;
   static float dfltVfov, dfltAspect, dfltNear, dfltFar; 
   static float dfltSpeed, dfltTurnRate;
   glm::mat4 lookMat;
};

#endif
