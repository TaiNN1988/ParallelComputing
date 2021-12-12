#define THRESHOLD 10000		/*Threshold */

/*A correct GPU implementation of Sobel filter*/
__kernel void KernelSobelFilter(
	__global unsigned char* d_outImage,
	__global const unsigned char* d_srcImage,
	int width,
	int height)
{
	const int row = get_global_id(0) + 1;
	const int col = get_global_id(1) + 1;

	if (col < (width - 1) && row < (height - 1)) /* check out of range */
	{
		// Computes image gradient horizontally
		int gradX = d_srcImage[(row - 1) * width + col + 1] - d_srcImage[(row - 1) * width + col - 1]
			+ 2 * d_srcImage[row * width + col + 1] - 2 * d_srcImage[row * width + col - 1]
			+ d_srcImage[(row + 1) * width + col + 1] - d_srcImage[(row + 1) * width + col - 1];

		// ComputsrcImageg gradient vertically
		int gradY = d_srcImage[(row - 1) * width + col - 1] + 2 * d_srcImage[(row - 1) * width + col] + d_srcImage[(row - 1) * width + col + 1]
			- d_srcImage[(row + 1) * width + col - 1] - 2 * d_srcImage[(row + 1) * width + col] - d_srcImage[(row + 1) * width + col + 1];

		// Overall gradient magnitude
		int magnitude = gradX * gradX + gradY * gradY;

		// Convert to black/white by comparing against some given threshold
		if (magnitude > THRESHOLD) {
			d_outImage[row * width + col] = 255;
		}
		else {
			d_outImage[row * width + col] = 0;
		}
	}
}