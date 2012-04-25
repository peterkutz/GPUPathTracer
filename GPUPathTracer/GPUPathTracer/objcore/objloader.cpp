////////////////////////////////////////////////////////////////////////////////////////////////////
// OBJCORE: A Simple Obj Library
// by Yining Karl Li
//
// objloader.cpp
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "objloader.h"
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <string.h>
#include "../glm/glm.hpp" 

using namespace std;

objLoader::objLoader(string filename, obj* newMesh){

	geomesh = newMesh;
	cout << "Loading OBJ File: " << filename << endl;
	ifstream fp_in;
	char * fname = (char*)filename.c_str();
	fp_in.open(fname);
	if(fp_in.is_open()){
        while (fp_in.good() ){
			string line;
            getline(fp_in,line);
            if(line.empty()){
                line="42";
            }
			istringstream liness(line);
			if(line[0]=='v' && line[1]=='t'){
				string v;
				string x;
				string y;
				string z;
				getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');
				geomesh->addTextureCoord(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='v' && line[1]=='n'){
				string v;
				string x;
				string y;
				string z;
				getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');
				geomesh->addNormal(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='v'){
				string v;
				string x;
				string y;
				string z;
				getline(liness, v, ' ');
				getline(liness, x, ' ');
				getline(liness, y, ' ');
				getline(liness, z, ' ');
				geomesh->addPoint(glm::vec3(::atof(x.c_str()), ::atof(y.c_str()), ::atof(z.c_str())));
			}else if(line[0]=='f'){
				string v;
				getline(liness, v, ' ');
				string delim1 = "//";
				string delim2 = "/";
				if(std::string::npos != line.find("//")){
					//std::cout << "Vertex-Normal Format" << std::endl;
					vector<int> pointList;
					vector<int> normalList;
					while(getline(liness, v, ' ')){
						istringstream facestring(v);
						string f;
						getline(facestring, f, '/');
						pointList.push_back(::atof(f.c_str())-1);

						getline(facestring, f, '/');
						getline(facestring, f, ' ');
						normalList.push_back(::atof(f.c_str())-1);

					}
					geomesh->addFace(pointList);
					geomesh->addFaceNormal(normalList);
				}else if(std::string::npos != line.find("/")){
					vector<int> pointList;
					vector<int> normalList;
					vector<int> texturecoordList;
					while(getline(liness, v, ' ')){
						istringstream facestring(v);
						string f;
						int i=0;
						while(getline(facestring, f, '/')){
							if(i==0){
								pointList.push_back(::atof(f.c_str())-1);
							}else if(i==1){
								texturecoordList.push_back(::atof(f.c_str())-1);
							}else if(i==2){
								normalList.push_back(::atof(f.c_str())-1);
							}
							i++;
						}
					}
					geomesh->addFace(pointList);
					geomesh->addFaceNormal(normalList);
					geomesh->addFaceTexture(texturecoordList);
				}else{
					string v;
					vector<int> pointList;
					while(getline(liness, v, ' ')){
						pointList.push_back(::atof(v.c_str())-1);
					}
					geomesh->addFace(pointList);
					//std::cout << "Vertex Format" << std::endl;
				}
			}
		}
		cout << "Loaded " << geomesh->getFaces()->size() << " faces, " << geomesh->getPoints()->size() << " vertices from " << filename << endl;
	}else{
        cout << "ERROR: " << filename << " could not be found" << endl;
    }
}

objLoader::~objLoader(){
}

obj* objLoader::getMesh(){
	return geomesh;
}
