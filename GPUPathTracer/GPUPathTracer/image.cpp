#include "image.h"
#include <iostream>

Image* newImage(int width, int height) {

	Image* image = new Image();
	image->width = width;
	image->height = height;
	image->numPixels = image->width * image->height;
	image->pixels = new Color[image->numPixels];
	image->passCounter = 0;
	image->startClock = clock();
	return image;

}

void deleteImage(Image* image) {

	if (image != NULL) {
		delete [] image->pixels;
		image->pixels = NULL;
		delete image;
		image = NULL;
	}

}

int pixelIndexRowColumn(Image* image, int i, int j) {
	return i * image->width + j;
}

Color& getPixelRowColumn(Image* image, int i, int j) {
	return image->pixels[pixelIndexRowColumn(image, i, j)];
}

void setPixelRowColumn(Image* image, int i, int j, Color c) {
	image->pixels[pixelIndexRowColumn(image, i, j)].x = c.x;
	image->pixels[pixelIndexRowColumn(image, i, j)].y = c.y;
	image->pixels[pixelIndexRowColumn(image, i, j)].z = c.z;
	//std::cout << c.x << " " << c.y << " " << c.z << std::endl;
}


float getSecondsElapsed(Image* image) {
	return ( (float)clock() - (float)image->startClock ) / (float)CLOCKS_PER_SEC;
}
float getFramesPerSecond(Image* image) {
	float secondsElapsed = getSecondsElapsed(image);
	if (secondsElapsed == 0) secondsElapsed = 1;
	return (float)image->passCounter / getSecondsElapsed(image);
}