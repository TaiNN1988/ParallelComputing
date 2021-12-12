#include "SobelFilter.h"
#include "CudaSobelFilter.h"
#include "CLSobelFilter.h"

#include <string>
#include <vector>
#include "../Helper/inc/Common.h"
#include <iostream>

#define TEST_DATA "./TestData/"

typedef int(*fun_ptr_sobelFilter)(unsigned char* outImage, unsigned char* inImage, int width, int height);
bool getConfig(fun_ptr_sobelFilter& out_fun_ptr, string& out_fileName, string& out_prefixName, int in_argc, char* in_argv[]);

// Utilities function
bool readImageFromPBMFile(unsigned char* &outImage, int& outWidth, int& outHeight, string PBM_fileName);
bool writeImageToPBMFile(unsigned char* srcImage, int width, int height, string PBM_fileName);

/*
cmd format: proram [CPU[SERIAL|PARALLEL]] | [GPU [CUDA|OPENCL]] [SOURCE_IMAGE_PPM_TYPE]

1_MulMatrixDemo.exe CPU SERIAL in1_valve.ppm
1_MulMatrixDemo.exe CPU PARALLEL in2_SEAsia_Still_ClimFatalities.ppm

1_MulMatrixDemo.exe GPU CUDA in2_SEAsia_Still_ClimFatalities.ppm
1_MulMatrixDemo.exe GPU OPENCL in3_airsCO2_printres0392.ppm
*/
int main(int argc, char* argv[])
{
	fun_ptr_sobelFilter fun_sobelFilter = nullptr;
	string fileName = "";
	string out_prefixName = "";
	bool err = getConfig(fun_sobelFilter, fileName, out_prefixName, argc, argv);
	if (!err)
	{
		printf("Error! Wrong cmd format. Should be: proram [CPU[S|P]] | [GPU [CUDA|OPENCL]] [SOURCE_IMAGE_PPM_TYPE]\n");
		return 0;
	}

	unsigned char* srcImage = NULL;
	unsigned char* dstImage = NULL;
	int width = 0;
	int height = 0;

	readImageFromPBMFile(srcImage, width, height,  TEST_DATA + fileName);

	cout << "File name of image: " << TEST_DATA + fileName << endl;
	cout << "Width of image: " << width << endl;
	cout << "Height of image: " << height << endl;

	/*Output modify image by execute sobel filter on CUDA*/
	dstImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));
	memset(dstImage, 0, width * height * sizeof(unsigned char));

	if (fun_sobelFilter != nullptr)
	{
		auto start = chrono::steady_clock::now();
		printf("========START measuring time=========\n");

		// Calculate matrix multiplication  outMatrix = inMatrix1 x inMatrix2
		fun_sobelFilter(dstImage, srcImage, width, height);

		printf("========END measuring time=========\n");
		auto end = chrono::steady_clock::now();
		auto diff = end - start;
		auto diff_miSec = chrono::duration_cast<chrono::milliseconds>(diff);
		//auto diff_miSec = chrono::duration_cast<chrono::microseconds>(diff);
		cout << "ExecutedTime = " << diff_miSec.count() << " (milliseconds)" << endl << endl;

		// Write ouput
		writeImageToPBMFile(dstImage, width, height, string(TEST_DATA) + string("out_") + out_prefixName + fileName);
	}

	free(srcImage);
	free(dstImage);
	return 0;
}

bool readImageFromPBMFile(unsigned char* &outImage, int& outWidth, int& outHeight, string PBM_fileName)
{
	bool err = true;
	FILE* src;
	unsigned char* srcImage;

	if (!(src = fopen(PBM_fileName.c_str(), "rb")))
	{
		printf("Couldn't open file %s for reading.\n", PBM_fileName.c_str());
		exit(1);
	}

	char p, s;
	fscanf(src, "%c%c\n", &p, &s);
	if (p != 'P' || s != '6')
	{
		printf("Not a valid PPM file (%c %c)\n", p, s);
		exit(1);
	}

	fscanf(src, "%d %d\n", &outWidth, &outHeight);

	int ignored;
	fscanf(src, "%d\n", &ignored);

	int pixels = outWidth * outHeight;
	srcImage = (unsigned char*)malloc(pixels * 3);
	if (fread(srcImage, sizeof(unsigned char), pixels * 3, src) != pixels * 3)
	{
		printf("Error reading file.\n");
		free(srcImage);
		exit(1);
	}
	fclose(src);

	int imageSize = outHeight * outWidth;
	outImage = (unsigned char*)malloc(imageSize * sizeof(unsigned char));

	//Convert input Image to grayScale
	for (int i = 0; i < imageSize; i++)
	{
		unsigned int r = srcImage[i * 3];
		unsigned int g = srcImage[i * 3 + 1];
		unsigned int b = srcImage[i * 3 + 2];
		outImage[i] = unsigned int(0.2989 * r + 0.5870 * g + 0.1140 * b);
	}
	return err;
}

bool writeImageToPBMFile(unsigned char* srcImage, int width, int height, string PBM_fileName)
{
	bool err = true;
	int imageSize = width * height;
	FILE* out;
	cout << "output image file: " << PBM_fileName << endl;

	if (!(out = fopen(PBM_fileName.c_str(), "wb")))
	{
		printf("Couldn't open file for output.\n");
		err = false;
	}
	fprintf(out, "P5\n%d %d\n255\n", width, height);
	if (fwrite(srcImage, sizeof(unsigned char), imageSize, out) != imageSize)
	{
		printf("Error writing file.\n");
		err = false;
	}
	fclose(out);
	return err;
}

bool getConfig(fun_ptr_sobelFilter& out_fun_ptr, string& out_fileName, string& out_prefixName, int in_argc, char* in_argv[])
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

	out_fileName = listArgv[2];

	if (listArgv[0] == "CPU") {
		if (listArgv[1] == "SERIAL") {
			out_fun_ptr = &CPU_sobelFilter_serial;
			out_prefixName = "CPU_SERIAL_";
			err = true;
		}
		else if (listArgv[1] == "PARALLEL") {
			out_fun_ptr = &CPU_sobelFilter_parallel;
			out_prefixName = "CPU_PARALLEL_";
			err = true;
		}
	}
	else if (listArgv[0] == "GPU") {
		if (listArgv[1] == "CUDA") {
			out_fun_ptr = &GPU_CUDA_sobelFilter;
			out_prefixName = "GPU_CUDA_";
			err = true;
		}
		else if (listArgv[1] == "OPENCL") {
			out_fun_ptr = &GPU_OPENCL_sobelFilter;
			out_prefixName = "CPU_OPENCL_";
			err = true;
		}
	}

	if (out_fun_ptr == nullptr && out_fileName.empty())
	{
		err = false;
	}
	return err;
}