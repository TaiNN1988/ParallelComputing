#include "MulMatrix.h"
#include "CudaMulMatrix.h"
#include "CLMulMatrix.h"
#include <string>
#include <vector>
#include "../Helper/inc/Common.h"

typedef void(*fun_ptr_mulMatrix)(int* outMatrix, int* inMatrix1, int* inMatrix2, int halfSize);
bool getConfig(fun_ptr_mulMatrix& out_fun_ptr, int& out_halfSize, int in_argc, char* in_argv[]);

/*
cmd format: proram [CPU[SERIAL|PARALLEL]] | [GPU [CUDA|OPENCL]] [SIZE]
SIZE = halfSize of Matrix

1_MulMatrixDemo.exe CPU SERIAL 1024
1_MulMatrixDemo.exe CPU PARALLEL 1024

1_MulMatrixDemo.exe GPU CUDA 1024
1_MulMatrixDemo.exe GPU OPENCL 1024
*/
int main(int argc, char* argv[])
{
	fun_ptr_mulMatrix fun_mulMatrix = nullptr;
	int halfSize = 0;
	bool err = getConfig(fun_mulMatrix, halfSize, argc, argv);
	if (!err)
	{
		printf("Error! Wrong cmd format. Should be: proram [CPU[S|P]] | [GPU [CUDA|OPENCL]] [SIZE]\n");
		return 0;
	}

	printf("halfSize = % d \n", halfSize);

	int numOfBytes = halfSize * halfSize * sizeof(int);
	int* inMatrix1 = (int*)malloc(numOfBytes);
	int* inMatrix2 = (int*)malloc(numOfBytes);
	int* outMatrix = (int*)malloc(numOfBytes);
	int* outMatrix_ori = (int*)malloc(numOfBytes);

	// Initial array
	srand(time(NULL));
	for (size_t i = 0; i < halfSize; i++)
	{
		for (size_t j = 0; j < halfSize; j++)
		{
			inMatrix1[j + i * halfSize] = rand() % 100 + 1;
			inMatrix2[j + i * halfSize] = rand() % 100 + 1;
			outMatrix[j + i * halfSize] = 0;
		}
	}

	cout << "Current function pointer: " << *fun_mulMatrix << endl;
	if (fun_mulMatrix != nullptr)
	{
		auto start = chrono::steady_clock::now();
		printf("========START measuring time=========\n");

		// Calculate matrix multiplication  outMatrix = inMatrix1 x inMatrix2
		fun_mulMatrix(outMatrix, inMatrix1, inMatrix2, halfSize);

		printf("========END measuring time=========\n");
		auto end = chrono::steady_clock::now();
		auto diff = end - start;
		auto diff_miSec = chrono::duration_cast<chrono::milliseconds>(diff);
		cout << "ExecutedTime = " << diff_miSec.count() << " (milliseconds)" << endl;

		// Verify output
		/*CPU_mulMatrix_parallel(outMatrix_ori, inMatrix1, inMatrix2, halfSize);
		bool isCompare = compareMatrix(outMatrix_ori, outMatrix, halfSize);
		printf("%s", isCompare ? "PASSED!" : "FAILED");*/

		/*printf("show 10x10\n");
		showMatrix(outMatrix, 10);*/
	}

	free(inMatrix1);
	free(inMatrix2);
	free(outMatrix);
	free(outMatrix_ori);

	return 0;
}

bool getConfig(fun_ptr_mulMatrix& out_fun_ptr, int& out_halfSize, int in_argc, char* in_argv[])
{
	bool err = false;

	if (in_argc != 4) {
		return err;
	}

	vector<string> listArgv;
	for (size_t i = 1; i < in_argc; i++)
	{
		listArgv.push_back(in_argv[i]);
	}

	out_halfSize = stoi(listArgv[2]);

	if (listArgv[0] == "CPU") {
		if (listArgv[1] == "SERIAL") {
			out_fun_ptr = &CPU_mulMatrix_serial;
			err = true;
		}
		else if (listArgv[1] == "PARALLEL") {
			out_fun_ptr = &CPU_mulMatrix_parallel;
			err = true;
		}
	}
	else if (listArgv[0] == "GPU") {
		if (listArgv[1] == "CUDA") {
			out_fun_ptr = &GPU_CUDA_mulMatrix;
			err = true;
		}
		else if (listArgv[1] == "OPENCL") {
			out_fun_ptr = &GPU_OPENCL_mulMatrix;
			err = true;
		}
	}

	if (out_fun_ptr == nullptr && out_halfSize < 8)
	{
		err = false;
	}
	return err;
}

void testMuMatrix_5_5()
{
	int in1[5][5] = {
		{1, 1, 1, 1, 1},
		{2, 2, 2, 2, 2},
		{3, 3, 3, 3, 3},
		{4, 4, 4, 4, 4},
		{5, 5, 5, 5, 5}
	};

	int in2[5][5] = {
		{1, 2, 3, 5, 1},
		{2, 2, 3, 5, 2},
		{3, 2, 3, 5, 3},
		{4, 2, 3, 5, 4},
		{5, 2, 3, 5, 5}
	};
	int out[5][5];
	int outExpected[5][5] = {
		{15, 10, 15, 25, 15},
		{30, 20, 30, 50, 30},
		{45, 30, 45, 75, 45},
		{60, 40, 60, 100, 60},
		{75, 50, 75, 125, 75}
	};

	int halfSize = 5;
	CPU_mulMatrix_serial(out[0], in1[0], in2[0], halfSize);

	// Compare
	for (size_t i = 0; i < halfSize; i++)
	{
		for (size_t j = 0; j < halfSize; j++)
		{
			if (out[i][j] != outExpected[i][j])
			{
				printf("Failed!");
				break;
			}
		}

	}
	printf("PASSED!");
}