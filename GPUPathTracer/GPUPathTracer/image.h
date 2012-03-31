#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"

struct Image {

	Color* pixels;
	int width;
	int height;
	int numPixels;

};



Image* newImage(int width, int height);
void deleteImage(Image* image);
int pixelIndexRowColumn(Image* image, int i, int j);
Color& getPixelRowColumn(Image* image, int i, int j);
void setPixelRowColumn(Image* image, int i, int j, Color c);

#endif // IMAGE_H