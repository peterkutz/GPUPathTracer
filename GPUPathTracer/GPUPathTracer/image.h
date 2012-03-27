#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"

struct Image {

	Color* pixels;
	int width;
	int height;
	int numPixels;

};


// Forward declarations:
Image* newImage(int width, int height);
void deleteImage(Image* image);


#endif // IMAGE_H