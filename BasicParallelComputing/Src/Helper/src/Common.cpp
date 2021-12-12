#include "../inc/Common.h"
#include <omp.h>

void showMatrix(int* matrix, int halfSize)
{
	for (size_t i = 0; i < halfSize; i++)
	{
		for (size_t j = 0; j < halfSize; j++)
		{
			printf("%d ", matrix[j + i * halfSize]);
		}
		printf("\n");
	}
}

bool compareMatrix(int* A, int* B, int halfSize)
{
	for (int i = 0; i < halfSize; i++)
	{
		for (int j = 0; j < halfSize; j++)
		{
			int idx = j + i * halfSize;
			if (A[idx] != B[idx])
				return false;
		}
	}
	return true;
}