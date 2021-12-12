#include "../Helper/inc/CudaHelper.h"
#include "CudaAddVector.h"


__global__ void KernelAddVector(int* d_outVector, int* d_inVector1, int* d_inVector2, int size)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < size)
	{
		d_outVector[i] = d_inVector1[i] + d_inVector2[i];
	}
}

/*
* Vector addition: outVector = inVector1 + inVector2
*/
int CudaAddVector(int* h_outVector, int* h_inVector1, int* h_inVector2, int numElements)
{
	int err = 0;
	int* d_outVector = NULL;
	int* d_inVector1 = NULL;
	int* d_inVector2 = NULL;
	int numOfByte = numElements * sizeof(int);

	// Allocate memory on GPU device
	err += CHECK_ERROR(cudaMalloc((void**)&d_outVector, numOfByte));
	err += CHECK_ERROR(cudaMalloc((void**)&d_inVector1, numOfByte));
	err += CHECK_ERROR(cudaMalloc((void**)&d_inVector2, numOfByte));

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(d_inVector1, h_inVector1, numOfByte, cudaMemcpyHostToDevice));
	err += CHECK_ERROR(cudaMemcpy(d_inVector2, h_inVector2, numOfByte, cudaMemcpyHostToDevice));

	// Setup thread, block to lauch kernel
	int threadsPerBlock = 256;
	int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	// Lauch kernal to calculate add vector on GPU
	KernelAddVector << <blocksPerGrid, threadsPerBlock >> > (d_outVector, d_inVector1, d_inVector2, numElements);

	err += CHECK_ERROR(cudaGetLastError());

	// Transfer data from host to device
	err += CHECK_ERROR(cudaMemcpy(h_outVector, d_outVector, numOfByte, cudaMemcpyDeviceToHost));

	// Free memory on device
	cudaFree(d_inVector1);
	cudaFree(d_inVector2);
	cudaFree(d_outVector);

	return err;
}
