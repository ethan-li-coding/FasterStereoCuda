#include "cusgm_util.cuh"
#include "cusgm_types.h"

#define THREADS_COMMON	128

namespace cusgm_util {
	__global__ void Kernel_ComputeDepth(float32* disp_map, sint32 dp_psize, sint32 width, sint32 height, float32* depth_left, CamParam_T cam_param)
	{
		//计算点的深度
		const sint32 image_x = blockIdx.x * blockDim.x + threadIdx.x;
		const sint32 image_y = blockIdx.y;
		float32* disp_ptr = (float32*)((uint8*)disp_map + image_y * dp_psize);
		float32* pDepth = (float32*)((uint8*)depth_left + image_y * dp_psize);
		if (image_x >= width) {
			return;
		}
		const float32 disp = disp_ptr[image_x];
		if (disp == INVALID_VALUE) {
			pDepth[image_x] = INVALID_VALUE;
			return;
		}
		const float64 x0_left = cam_param.x0_left;
		const float64 y0_left = cam_param.y0_left;
		const float64 x0_right = cam_param.x0_right;
		const float64 f_left = cam_param.focus;
		const float64 baseline = cam_param.baseline;

		const float64 factor1 = baseline * f_left;
		const float64 factor2 = x0_right - x0_left;

		const float64 depth = factor1 / (disp + factor2);

		pDepth[image_x] = static_cast<float32>(depth);
	}
 
	void ComputeDepthCuda(float32* disp_map, sint32 dp_psize, sint32 width, sint32 height, float32* depth_left, CamParam_T cam_param)
	{
		const sint32 threadsize = THREADS_COMMON;

		dim3 block(ceil(width * 1.0 / threadsize), height);
		Kernel_ComputeDepth << <block, threadsize >> > (disp_map, dp_psize, width, height, depth_left, cam_param);
#ifdef SYNCHRONIZE
		cudaDeviceSynchronize();
#endif
	}

	__global__ void Kernel_ComputeRightInitialValue(sint16* init_val_left, sint16* init_val_right, sint32 width, sint32 height, size_t pitch_size)
	{
		sint16* left = (sint16*)((uint8*)init_val_left + blockIdx.y * pitch_size);
		sint16* right = (sint16*)((uint8*)init_val_right + blockIdx.y * pitch_size);
		const sint16 image_xl = blockIdx.x * blockDim.x + threadIdx.x;
		if (image_xl >= width) return;
		right[image_xl] = INVALID_VALUE_SHORT;
		__syncthreads();

		sint16 init_l = left[image_xl] + (LEVEL_RANGE >> 1);
		if (image_xl - init_l > 0 && image_xl - init_l < width)
			right[image_xl - init_l] = init_l - (LEVEL_RANGE >> 1);
	}
	void ComputeRightInitialValue(sint16* init_val_left, sint16* init_val_right, sint32 width, sint32 height, size_t pitch_size)
	{
		const sint32 threadsize = THREADS_COMMON;

		dim3 threads(threadsize, 1);
		dim3 block(ceil(width * 1.0 / threadsize), height);

		Kernel_ComputeRightInitialValue << <block, threads >> > (init_val_left, init_val_right, width, height, pitch_size);

#ifdef SYNCHRONIZE
		cudaDeviceSynchronize();
#endif
	}
}
