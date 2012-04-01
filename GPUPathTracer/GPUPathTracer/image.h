#ifndef IMAGE_H
#define IMAGE_H

#include "color.h"

// TODO: Make this a class!!! (It's no longer used in the .cu file!)

struct Image {

	Color* pixels;
	int width;
	int height;
	int numPixels;
	int passCounter;

};



Image* newImage(int width, int height);
void deleteImage(Image* image);
int pixelIndexRowColumn(Image* image, int i, int j);
Color& getPixelRowColumn(Image* image, int i, int j);
void setPixelRowColumn(Image* image, int i, int j, Color c);

#endif // IMAGE_H