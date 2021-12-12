#include "../Helper/inc/CLHelper.h"
#include "CLSobelFilter.h"

#define MAX_SOURCE_SIZE (0x100000)
#define KERNEL_FILE_NAME "KernelSobelFilter.cl"
#define KERNEL_FUNCTION_NAME "KernelSobelFilter"

int GPU_OPENCL_sobelFilter(unsigned char* outImage, unsigned char* srcImage, int width, int height)
{
	printf("Invoked GPU_OPENCL_sobelFilter function\n");
	int numOfByte = width * height * sizeof(unsigned char);

	// Load kernel from file vecAddKernel.cl
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;

	kernelFile = fopen(KERNEL_FILE_NAME, "r");

	if (!kernelFile) {

		fprintf(stderr, "No file named KernelMulMatrix.cl was found\n");

		//exit(-1);

	}
	kernelSource = (char*)malloc(MAX_SOURCE_SIZE);
	kernelSize = fread(kernelSource, 1, MAX_SOURCE_SIZE, kernelFile);
	fclose(kernelFile);

	// Getting platform and device information
	cl_platform_id platformId = NULL;
	cl_device_id deviceID = NULL;
	cl_uint retNumDevices;
	cl_uint retNumPlatforms;
	cl_int ret = CHECK_ERROR(clGetPlatformIDs(1, &platformId, &retNumPlatforms));
	ret = clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices);

	// print device name
	/*char* value;
	size_t valueSize;
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);*/

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);

	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem d_srcImage = clCreateBuffer(context, CL_MEM_READ_ONLY, numOfByte, NULL, &ret);
	cl_mem d_outImage = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numOfByte, NULL, &ret);
	 cl_int d_width = width;
	 cl_int d_height = height;


	// Copy lists to memory buffers
	ret = CHECK_ERROR(clEnqueueWriteBuffer(commandQueue, d_srcImage, CL_TRUE, 0, numOfByte, srcImage, 0, NULL, NULL));

	// Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	// Build program
	ret = CHECK_ERROR(clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL));
	if (ret == CL_BUILD_PROGRAM_FAILURE) { // If compile failed, print the error message
	 // Determine the size of the log
		size_t log_size;
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		char* log = (char*)malloc(log_size);

		// Get the log and print it
		clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
		printf("%s\n", log);
	}

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, KERNEL_FUNCTION_NAME, &ret);

	// Set arguments for kernel
	ret = CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_outImage));
	ret = CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_srcImage));
	ret = CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&d_width));
	ret = CHECK_ERROR(clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&d_height));


	// Execute the kernel
	//size_t globalItemSize = size;
	//size_t localItemSize = 64; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16 
	const int TS = 8;
	const size_t local[] = { TS, TS,1 };
	const size_t global[] = { height, width,1 };

	ret = CHECK_ERROR(clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, global, local, 0, NULL, NULL));
	if (ret != 0)
	{
		printf("error: %d\n", ret);
	}

	// Read from device back to host.
	ret = CHECK_ERROR(clEnqueueReadBuffer(commandQueue, d_outImage, CL_TRUE, 0, numOfByte, outImage, 0, NULL, NULL));

	// Clean up, release memory.
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(d_srcImage);
	ret = clReleaseMemObject(d_outImage);
	ret = clReleaseContext(context);
	return ret;
}