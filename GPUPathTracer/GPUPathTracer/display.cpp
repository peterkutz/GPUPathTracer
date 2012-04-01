#include "windows_include.h"

// includes, system
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <iostream>

// includes, GL
#include <GL/glew.h>

// includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "cutil_math.h"

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "path_tracer.h"
#include "image.h"
#include "camera.h"
#include "basic_math.h"




////////////////////////////////////////////////////////////////////////////////
// constants

// window
int window_width = 512; // Currently not related to render width.
int window_height = 512; // Currently not related to render height.

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -30.0;

PathTracer pathTracer;
Camera theCamera;

////////////////////////////////////////////////////////////////////////////////
// forward declarations
void initializeThings( int argc, char** argv);

// GL functionality
bool initGL();
void createVBOs( GLuint* vbo);
void deleteVBOs( GLuint* vbo);

// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {

    initializeThings( argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
// Initialize camera.
////////////////////////////////////////////////////////////////////////////////

void initCamera()
{
   double w = 1;   
   double h = 1;   
   double d = 1;   
   double angle = 0.5*theCamera.dfltVfov*BasicMath::PI/180.0;
   double dist;
   if (w > h) dist = w*0.5/std::tan(angle);  // aspect is 1, so i can do this
   else dist = h*0.5/std::tan(angle);
  // theCamera.dfltEye = glm::vec3(w*0.5, h*0.5, -(dist+d*0.5));

   theCamera.dfltEye = glm::vec3(0,0,4.8);
   //theCamera.dfltLook = glm::vec3(w*0.5, h*0.5, 0.0);
   theCamera.dfltLook = glm::vec3(0, 0, 0.0);
   theCamera.reset();
}

////////////////////////////////////////////////////////////////////////////////
// Initialize things.
////////////////////////////////////////////////////////////////////////////////

void initializeThings( int argc, char** argv) {
	
    // Init random number generator
	srand((unsigned)time(0));
	initCamera();

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
	if (rand() < RAND_MAX / 2) {
		glutCreateWindow( "Peter and Karl's GPU Path Tracer");
	} else {
		glutCreateWindow( "Karl and Peter's GPU Path Tracer");
	}

    // initialize GL
    if( false == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);


    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
// Initialize GL
////////////////////////////////////////////////////////////////////////////////
bool initGL() {

    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return false;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // Set up an orthographic view:
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0,1,0,1,-1,1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();


	// Create a texture for displaying the render:
	// TODO: Move to a function.
	glBindTexture(GL_TEXTURE_2D, 13);
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	// Use nearest-neighbor point sampling instead of linear interpolation:
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); //glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);

	// Enable textures:
	glEnable(GL_TEXTURE_2D);


    //CUT_CHECK_ERROR_GL(); // ???

    return true;
}



////////////////////////////////////////////////////////////////////////////////
// Display callback
////////////////////////////////////////////////////////////////////////////////
void display() {

	theCamera.buildRenderCam(pathTracer.rendercam);

	Image* imageReference = pathTracer.render(); 

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// Update the texture:
	int imageWidth = imageReference->width;
	int imageHeight = imageReference->height;
	uchar3* imageData = new uchar3[imageWidth * imageHeight];
	for (int i = 0; i < imageHeight; i++) {
		for (int j = 0; j < imageWidth; j++) {
			Color c = getPixelRowColumn(imageReference, i, j);
			imageData[pixelIndexRowColumn(imageReference, i, j)] = floatTo8Bit(c / imageReference->passCounter);
		}
	}


	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGB, imageWidth, imageHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, imageData);
	delete [] imageData; // glTexImage2D makes a copy of the data, so the original data can (and should!) be deleted here (otherwise it will leak memory like a madman).


	// Show the texture:
	glBindTexture (GL_TEXTURE_2D, 13);
	glBegin (GL_QUADS);
	glTexCoord2f (0.0, 0.0);
	glVertex3f (0.0, 1.0, 0.0);
	glTexCoord2f (1.0, 0.0);
	glVertex3f (1.0, 1.0, 0.0);
	glTexCoord2f (1.0, 1.0);
	glVertex3f (1.0, 0.0, 0.0);
	glTexCoord2f (0.0, 1.0);
	glVertex3f (0.0, 0.0, 0.0);
	glEnd ();



	//write the iteration count to the display
//	glPushAttrib(GL_LIGHTING_BIT);
//		glDisable(GL_LIGHTING);
//		glColor4f(1.0, 1.0, 1.0, 1.0);
//		glMatrixMode(GL_PROJECTION);
//		glLoadIdentity();
//		gluOrtho2D(0.0, 1.0, 0.0, 1.0);
//
//		glMatrixMode(GL_MODELVIEW);
//		glLoadIdentity();

		glColor4f(0.0, 0.0, 0.0, 0.0); // glColor4f affects the texture display color and vice versa. Set it to black here.

		glRasterPos2f(.01, 0.01); 
  
		char info[1024];
		sprintf(info, "Iterations: %u", imageReference->passCounter);
		for (unsigned int i = 0; i < strlen(info); i++){
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, info[i]);
		}

		glColor4f(1.0, 1.0, 1.0, 1.0); // glColor4f affects the texture display color and vice versa, so reset it to white here.
//	glPopAttrib();



    glutSwapBuffers();
    glutPostRedisplay();
}

////////////////////////////////////////////////////////////////////////////////
// Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        exit( 0);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
/*void mouse(int button, int state, int x, int y) {

    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y) {

    float dx, dy;
    dx = (float)(x - mouse_old_x);
    dy = (float)(y - mouse_old_y);

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2f;
        rotate_y += dx * 0.2f;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01f;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}*/

int lastX = 0, lastY = 0;
int theButtonState = 0;
int theModifierState = 0;

void motion(int x, int y)
{
   int deltaX = lastX - x;
   int deltaY = lastY - y;
   bool moveLeftRight = abs(deltaX) > abs(deltaY);
   bool moveUpDown = !moveLeftRight;

   if (theButtonState == GLUT_LEFT_BUTTON)  // Rotate
   {
      if (moveLeftRight && deltaX > 0) theCamera.orbitLeft(deltaX);
      else if (moveLeftRight && deltaX < 0) theCamera.orbitRight(-deltaX);
      else if (moveUpDown && deltaY > 0) theCamera.orbitUp(deltaY);
      else if (moveUpDown && deltaY < 0) theCamera.orbitDown(-deltaY);
   }
   else if (theButtonState == GLUT_MIDDLE_BUTTON) // Zoom
   {
      if (moveUpDown && deltaY > 0) theCamera.moveForward(deltaY);
      else if (moveUpDown && deltaY < 0) theCamera.moveBack(-deltaY);
   }    

   if (theModifierState & GLUT_ACTIVE_ALT) // camera move
   {
      if (theButtonState == GLUT_RIGHT_BUTTON) // Pan
      {
         if (moveLeftRight && deltaX > 0) theCamera.moveLeft(deltaX);
         else if (moveLeftRight && deltaX < 0) theCamera.moveRight(-deltaX);
         else if (moveUpDown && deltaY > 0) theCamera.moveUp(deltaY);
         else if (moveUpDown && deltaY < 0) theCamera.moveDown(-deltaY);
      }   
   }
 
   lastX = x;
   lastY = y;
   pathTracer.Reset();
   glutPostRedisplay();
}

void mouse(int button, int state, int x, int y)
{
   theButtonState = button;
   theModifierState = glutGetModifiers();
   lastX = x;
   lastY = y;

   //glutSetMenu(theMenu);
   //if (theModifierState & GLUT_ACTIVE_ALT)
   //{
    //  glutDetachMenu(GLUT_RIGHT_BUTTON);
   //}
   //else
  // {
   //   glutAttachMenu(GLUT_RIGHT_BUTTON);
   //}

   motion(x, y);
}