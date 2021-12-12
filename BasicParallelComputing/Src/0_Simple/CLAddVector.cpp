#include "../Helper/inc/CLHelper.h"
#include "CLAddVector.h"

#define MAX_SOURCE_SIZE (0x100000)
#define KERNEL_FILE_NAME "KernelAddVector.cl"
#define KERNEL_FUNCTION_NAME "addVectors"

int clAddVector(int* h_outVector, int* h_inVector1, int* h_inVector2, int numElements)
{
	int numOfByte = numElements * sizeof(int);

	// Load kernel from file vecAddKernel.cl
	FILE* kernelFile;
	char* kernelSource;
	size_t kernelSize;

	kernelFile = fopen(KERNEL_FILE_NAME, "r");

	if (!kernelFile) {

		fprintf(stderr, "No file named vecAddKernel.cl was found\n");

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
	ret = CHECK_ERROR(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_DEFAULT, 1, &deviceID, &retNumDevices));

	// print device name
	char* value;
	size_t valueSize;
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, 0, NULL, &valueSize);
	value = (char*)malloc(valueSize);
	clGetDeviceInfo(deviceID, CL_DEVICE_NAME, valueSize, value, NULL);
	printf("Device: %s\n", value);
	free(value);

	// Creating context.
	cl_context context = clCreateContext(NULL, 1, &deviceID, NULL, NULL, &ret);

	// Creating command queue
	cl_command_queue commandQueue = clCreateCommandQueue(context, deviceID, 0, &ret);

	// Memory buffers for each array
	cl_mem d_inVector1 = clCreateBuffer(context, CL_MEM_READ_ONLY, numOfByte, NULL, &ret);
	cl_mem d_inVector2 = clCreateBuffer(context, CL_MEM_READ_ONLY, numOfByte, NULL, &ret);
	cl_mem d_outVector = clCreateBuffer(context, CL_MEM_WRITE_ONLY, numOfByte, NULL, &ret);

	// Copy lists to memory buffers
	ret = CHECK_ERROR(clEnqueueWriteBuffer(commandQueue, d_inVector1, CL_TRUE, 0, numOfByte, h_inVector1, 0, NULL, NULL));
	ret = CHECK_ERROR(clEnqueueWriteBuffer(commandQueue, d_inVector2, CL_TRUE, 0, numOfByte, h_inVector2, 0, NULL, NULL));

	// Create program from kernel source
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSource, (const size_t*)&kernelSize, &ret);

	// Build program
	ret = CHECK_ERROR(clBuildProgram(program, 1, &deviceID, NULL, NULL, NULL));

	// Create kernel
	cl_kernel kernel = clCreateKernel(program, "addVectors", &ret);

	// Set arguments for kernel
	ret = CHECK_ERROR(clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&d_outVector));
	ret = CHECK_ERROR(clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&d_inVector1));
	ret = CHECK_ERROR(clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_inVector2));

	// Execute the kernel
	size_t globalItemSize = numElements;
	size_t localItemSize = 64; // globalItemSize has to be a multiple of localItemSize. 1024/64 = 16 
	ret = CHECK_ERROR(clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &globalItemSize, &localItemSize, 0, NULL, NULL));

	// Read from device back to host.
	ret = CHECK_ERROR(clEnqueueReadBuffer(commandQueue, d_outVector, CL_TRUE, 0, numOfByte, h_outVector, 0, NULL, NULL));

	// Clean up, release memory.
	ret = clFlush(commandQueue);
	ret = clFinish(commandQueue);
	ret = clReleaseCommandQueue(commandQueue);
	ret = clReleaseKernel(kernel);
	ret = clReleaseProgram(program);
	ret = clReleaseMemObject(d_inVector1);
	ret = clReleaseMemObject(d_inVector2);
	ret = clReleaseMemObject(d_outVector);
	ret = clReleaseContext(context);

	return 0;
}