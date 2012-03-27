#include "image.h"


Image* newImage(int width, int height) {

	Image* image = new Image();
	image->width = width;
	image->height = height;
	image->numPixels = image->width * image->width;
	image->pixels = new Color[image->numPixels];
	return image;

}

void deleteImage(Image* image) {

	delete [] image->pixels;
	image->pixels = NULL;
	delete image;
	image = NULL;

}

