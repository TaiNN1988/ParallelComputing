#include "../Helper/inc/Common.h"
#include "SobelFilter.h"
#include <omp.h>

#define THRESHOLD 10000		/*Threshold */

int CPU_sobelFilter_serial(unsigned char* outImage, unsigned char* srcImage, int width, int height)
{
	printf("Invoked CPU_sobelFilter_serial function\n");
	int gradX, gradY, magnitude;

	for (int row = 1; row < height - 1; row++)
	{
		for (int col = 1; col < width - 1; col++)
		{
			// Computes image gradient horizontally
			gradX = srcImage[(row - 1) * width + col + 1] - srcImage[(row - 1) * width + col - 1]
				+ 2 * srcImage[row * width + col + 1] - 2 * srcImage[row * width + col - 1]
				+ srcImage[(row + 1) * width + col + 1] - srcImage[(row + 1) * width + col - 1];

			// Computing gradient vertically
			gradY = srcImage[(row - 1) * width + col - 1] + 2 * srcImage[(row - 1) * width + col] + srcImage[(row - 1) * width + col + 1]
				- srcImage[(row + 1) * width + col - 1] - 2 * srcImage[(row + 1) * width + col] - srcImage[(row + 1) * width + col + 1];

			// Overall gradient magnitude
			magnitude = gradX * gradX + gradY * gradY;

			if (magnitude > THRESHOLD){
				outImage[row * width + col] = 255;
			}
			else {
				outImage[row * width + col] = 0;
			}
		}
	}
	return 0;
}


int CPU_sobelFilter_parallel(unsigned char* outImage, unsigned char* srcImage, int width, int height)
{
	printf("Invoked CPU_sobelFilter_parallel function\n");

#pragma omp parallel for
	for (int row = 1; row < height - 1; row++)
	{
		for (int col = 1; col < width - 1; col++)
		{
			// Computes image gradient horizontally
			int gradX = srcImage[(row - 1) * width + col + 1] - srcImage[(row - 1) * width + col - 1]
				+ 2 * srcImage[row * width + col + 1] - 2 * srcImage[row * width + col - 1]
				+ srcImage[(row + 1) * width + col + 1] - srcImage[(row + 1) * width + col - 1];

			// Computing gradient vertically
			int gradY = srcImage[(row - 1) * width + col - 1] + 2 * srcImage[(row - 1) * width + col] + srcImage[(row - 1) * width + col + 1]
				- srcImage[(row + 1) * width + col - 1] - 2 * srcImage[(row + 1) * width + col] - srcImage[(row + 1) * width + col + 1];

			// Overall gradient magnitude
			int magnitude = gradX * gradX + gradY * gradY;

			if (magnitude > THRESHOLD) {
				outImage[row * width + col] = 255;
			}
			else {
				outImage[row * width + col] = 0;
			}
		}
	}
	return 0;
}
