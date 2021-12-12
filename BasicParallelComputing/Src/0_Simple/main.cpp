#include "../Helper/inc/Common.h"
#include "AddVector.h"
#include "CudaAddVector.h"
#include "CLAddVector.h"

#define CPU_ADD_VECTOR_SERIAL_CODE 0
#define CPU_ADD_VECTOR_PARALLEL_CODE 0
#define GPU_ADD_VECTOR_CUDA 0
#define GPU_ADD_VECTOR_OPENCL 1


int main(int argc, char* argv[])
{
	int size = 100000000;
	int* in1 = (int*)malloc(size * sizeof(int));
	int* in2 = (int*)malloc(size * sizeof(int));
	int* out = (int*)malloc(size * sizeof(int));

	// Init data
	srand(time(NULL));
	for (int i = 0; i < size; i++)
	{
		in1[i] = rand() % 1000 + 1;
		in2[i] = rand() % 1000 + 1;
	}

	auto start = chrono::steady_clock::now();
	printf("Start...\n");

#if CPU_ADD_VECTOR_SERIAL_CODE == 1
	printf("Execute <<addVector function>>\n");
	addVector(out, in1, in2, size);

#elif CPU_ADD_VECTOR_PARALLEL_CODE == 1
	printf("Execute <<addVectorParallel function>>\n");
	addVectorParallel(out, in1, in2, size);

#elif GPU_ADD_VECTOR_CUDA == 1
	printf("Execute <<CudaAddVector function>>\n");
	CudaAddVector(out, in1, in2, size);

#elif GPU_ADD_VECTOR_OPENCL == 1
	printf("Execute <<clAddVector function>>\n");
	clAddVector(out, in1, in2, size);
#endif 

	auto end = chrono::steady_clock::now();
	printf("End\n");
	auto diff = end - start;
	auto diff_miSec = chrono::duration_cast<chrono::milliseconds>(diff);
	cout << "ExecutedTime = " << diff_miSec.count() << endl;

	// Compare
	for (int i = 0; i < size; i++)
	{
		int ori = in1[i] + in2[i];
		if (out[i] != ori) {
			printf("Failed!");
			break;
		}
	}
	printf("PASSED!");

	free(in1);
	free(in2);
	free(out);

	return 0;
}