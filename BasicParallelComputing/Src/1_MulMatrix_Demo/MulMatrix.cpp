#include "../Helper/inc/Common.h"
#include "MulMatrix.h"
#include <omp.h>

// out = A*B
void CPU_mulMatrix_serial(int* outMatrix, int* inMatrixA, int* inMatrixB, int halfSize)
{
	printf("Invoked CPU_mulMatrix_serial function\n");
	for (int i = 0; i < halfSize; i++)
	{
		for (int j = 0; j < halfSize; j++)
		{
			int sum = 0;

			for (int k = 0; k < halfSize; k++)
			{
				sum = sum + inMatrixA[k + i * halfSize] * inMatrixB[j + k * halfSize];
			}
			outMatrix[j + i * halfSize] = sum;
		}
	}
}


// out = A*B
void CPU_mulMatrix_parallel(int* outMatrix, int* inMatrixA, int* inMatrixB, int halfSize)
{
	printf("Invoked CPU_mulMatrix_parallel function\n");
#pragma omp parallel for
	for (int i = 0; i < halfSize; i++)
	{
		/*if (i == 0)
		{
			printf("Threads: %d\n", omp_get_num_threads());
		}*/
		for (int j = 0; j < halfSize; j++)
		{
			int sum = 0;
			for (int k = 0; k < halfSize; k++)
			{
				sum = sum + inMatrixA[k + i * halfSize] * inMatrixB[j + k * halfSize];
			}
			outMatrix[j + i * halfSize] = sum;
		}
	}

}
