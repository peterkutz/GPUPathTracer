////////////////////////////////////////////////////////////////////////////////////////////////////
// OBJCORE: A Simple Obj Library
// by Yining Karl Li
//
// obj.h
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OBJ
#define OBJ

#include "../glm/glm.hpp"
#include <string>
#include <vector>

using namespace std;

class obj{
private:
	vector<glm::vec4> points;
	vector<vector<int> > faces; 
	vector<vector<int> > facenormals; 
	vector<vector<int> > facetextures; 
    vector<float*> faceboxes;   //bounding boxes for each face are stored in vbo-format!
	vector<glm::vec4> normals;
	vector<glm::vec4> texturecoords;
	int vbosize;
	int nbosize;
	int cbosize;
	int ibosize;
	float* vbo;
	float* nbo;
	float* cbo;
	unsigned short* ibo;
	float* boundingbox;
	float top;
	glm::vec3 defaultColor;
	bool maxminSet;
	float xmax; float xmin; float ymax; float ymin; float zmax; float zmin; 
public:
	obj();
	~obj();  

	//-------------------------------
	//-------Mesh Operations---------
	//-------------------------------
	void buildVBOs();
	void addPoint(glm::vec3);
	void addFace(vector<int>);
	void addNormal(glm::vec3);
	void addTextureCoord(glm::vec3);
	void addFaceNormal(vector<int>);
	void addFaceTexture(vector<int>);
	void compareMaxMin(float, float, float);
	bool isConvex(vector<int>);
	void recenter();

	//-------------------------------
	//-------Get/Set Operations------
	//-------------------------------
	float* getBoundingBox();    //returns vbo-formatted bounding box
	float getTop();
	void setColor(glm::vec3);
	glm::vec3 getColor();
	float* getVBO();
	float* getCBO();
	float* getNBO();
	unsigned short* getIBO();
	int getVBOsize();
	int getNBOsize();
	int getIBOsize();
	int getCBOsize();
    vector<glm::vec4>* getPoints();
	vector<vector<int> >* getFaces(); 
	vector<vector<int> >* getFaceNormals(); 
	vector<vector<int> >* getFaceTextures(); 
	vector<glm::vec4>* getNormals();
	vector<glm::vec4>* getTextureCoords();
    vector<float*>* getFaceBoxes();
	int GetNumTris();
	glm::vec3 getMin();
	glm::vec3 getMax();
};

#endif