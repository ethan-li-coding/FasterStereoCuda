/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li <ethan.li.whu@gmail.com>
* Describe	: header of cusgm_types
*/

#ifndef SGM_CUDA_TYPES_H_
#define SGM_CUDA_TYPES_H_
#include "cuda_runtime.h"
#include <device_launch_parameters.h>
#include <cstdio>

#define LEVEL_RANGE 32
#define PATH_NUM 4

#define INVALID_COST		127
#define INVALID_VALUE_SHORT -32768

#define CU_PI 3.1415926535

#ifndef CU_MINMAX
#define CU_MINMAX
#define cu_min(a,b) (((a)<(b))?(a):(b))
#define cu_max(a,b) (((a)>(b))?(a):(b))
#endif

inline bool CudaSafeCall(cudaError err)
{
	if (err != cudaSuccess ) {
		printf("Runtime API Error: %s.\n", cudaGetErrorString(err));
		return false;
	}
	else {
		return true;
	}
}

#define SafeCudaFree(Ptr) { if(Ptr) cudaFree(Ptr); Ptr = nullptr; }

inline void safeFree3D(cudaPitchedPtr* ptr)
{
	if (ptr) { if (ptr->ptr) { (cudaFree(ptr->ptr)); ptr->ptr = nullptr; } }
}

#endif