////////////////////////////////////////////////////////////////////////////////////////////////////
// OBJCORE: A Simple Obj Library
// by Yining Karl Li
//
// obj.h
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef OBJLOADER
#define OBJLOADER

#include <stdlib.h>
#include "obj.h"

using namespace std;

class objLoader{
private:
	obj* geomesh;
public:
	objLoader(string, obj*);
	~objLoader();
    
    //------------------------
    //-------GETTERS----------
    //------------------------
    
	obj* getMesh();
};

#endif