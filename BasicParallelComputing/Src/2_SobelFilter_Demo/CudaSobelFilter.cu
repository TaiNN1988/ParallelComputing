#include "../Helper/inc/CudaHelper.h"
#include "CudaSobelFilter.h"

#define THREADSPERBLOCK 8	/*Number of thread per block*/
#define THRESHOLD 10000		/*Threshold */

/*A correct GPU implementation of Sobel filter*/
__global__ void KernelSobelFilter(unsigned char* d_outImage, unsigned char* d_srcImage, int width, int height)
{
	int row = threadIdx.x + blockIdx.x * blockDim.x + 1;
	int col = threadIdx.y + blockIdx.y * blockDim.y + 1;

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

int GPU_CUDA_sobelFilter(unsigned char* outImage, unsigned char* srcImage, int width, int height)
{
	printf("Invoked GPU_CUDA_mulMatrix function\n");
	int err = 0;
	unsigned char* d_srcImage = NULL;
	unsigned char* d_outImage = NULL;
	int numOfByte = width * height * sizeof(unsigned char);

	// Allocate memory on GPU device
	err += CHECK_ERROR(cudaMalloc((void**)&d_srcImage, numOfByte));
	err += CHECK_ERROR(cudaMalloc((void**)&d_outImage, numOfByte));

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(d_srcImage, srcImage, numOfByte, cudaMemcpyHostToDevice));

	err += CHECK_ERROR(cudaMemset(d_outImage, 0, numOfByte));

	// Setup thread, block to lauch kernel
	dim3 threads(THREADSPERBLOCK, THREADSPERBLOCK);
	dim3 gridSize((height + threads.x - 1) / threads.x, (width + threads.y - 1) / threads.y);

	// Lauch kernal to calculate add vector on GPU
	KernelSobelFilter << <gridSize, threads >> > (d_outImage, d_srcImage, width, height);

	err += CHECK_ERROR(cudaGetLastError());

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(outImage, d_outImage, numOfByte, cudaMemcpyDeviceToHost));

	// Free memory on device
	cudaFree(d_srcImage);
	cudaFree(d_outImage);
	return err;
}