#ifndef SGM_CUDA_UTIL_H_
#define SGM_CUDA_UTIL_H_
#include "sgm_types.h"

namespace cusgm_util {
	/** \brief 计算深度 */
	void ComputeDepthCuda(float32* disp_map, sint32 dp_psize, sint32 width, sint32 height, float32* depth_left, CamParam_T cam_param);
	
	/** \brief 估计右初始视差图 */
	void ComputeRightInitialValue(sint16* init_val_left, sint16* init_val_right, sint32 width, sint32 height, size_t pitch_size);
}

#endif