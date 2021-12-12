#pragma once
#include "cuda_runtime.h"
#include <stdio.h>

/*****************************************************************
*Content CheckError:  This will output the proper CUDA error strings in the event
*					  that a CUDA host call returns an error
*
*Return: int  error code
*
******************************************************************/
inline int CheckError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
	}
	return err;
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))
