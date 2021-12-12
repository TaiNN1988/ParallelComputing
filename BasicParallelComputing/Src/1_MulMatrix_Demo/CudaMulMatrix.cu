#include "../Helper/inc/CudaHelper.h"
#include "CudaMulMatrix.h"

#define THREADSPERBLOCK 8	/*Number of thread per block*/

__global__ void KernelMulMatrix(int* d_outMatrix, int* d_inMatrix1, int* d_inMatrix2, int halfSize)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	if (i < halfSize && j < halfSize)
	{
		int sum = 0;
		for (int k = 0; k < halfSize; k++)
		{
			sum = sum + d_inMatrix1[k + i * halfSize] * d_inMatrix2[j + k * halfSize];
		}

		d_outMatrix[j + i * halfSize] = sum;
	}
}

/*
* Vector addition: outVector = inVector1 + inVector2
*/
void GPU_CUDA_mulMatrix(int* h_outMatrix, int* h_inMatrix1, int* h_inMatrix2, int halfSize)
{
	printf("Invoked GPU_CUDA_mulMatrix function\n");
	int err = 0;
	int* d_outMatrix = NULL;
	int* d_inMatrix1 = NULL;
	int* d_inMatrix2 = NULL;
	int numOfByte = halfSize * halfSize * sizeof(int);

	// Allocate memory on GPU device
	err += CHECK_ERROR(cudaMalloc((void**)&d_outMatrix, numOfByte));
	err += CHECK_ERROR(cudaMalloc((void**)&d_inMatrix1, numOfByte));
	err += CHECK_ERROR(cudaMalloc((void**)&d_inMatrix2, numOfByte));

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(d_inMatrix1, h_inMatrix1, numOfByte, cudaMemcpyHostToDevice));
	err += CHECK_ERROR(cudaMemcpy(d_inMatrix2, h_inMatrix2, numOfByte, cudaMemcpyHostToDevice));

	// Setup thread, block to lauch kernel
	dim3 threads(THREADSPERBLOCK, THREADSPERBLOCK);
	dim3 gridSize(halfSize / threads.x, halfSize / threads.y);

	// Lauch kernal to calculate add vector on GPU
	KernelMulMatrix << <gridSize, threads >> > (d_outMatrix, d_inMatrix1, d_inMatrix2, halfSize);

	err += CHECK_ERROR(cudaGetLastError());

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(h_outMatrix, d_outMatrix, numOfByte, cudaMemcpyDeviceToHost));

	// Free memory on device
	cudaFree(d_inMatrix1);
	cudaFree(d_inMatrix2);
	cudaFree(d_outMatrix);
}
