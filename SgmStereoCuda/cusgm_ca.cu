#include "cusgm_ca.cuh"
#include "cusgm_types.h"

#define THREADS_COMMON	128

namespace cusgm_ca{

__device__ inline uint8* GetCostPtr(cudaPitchedPtr cost, sint32 x, sint32 y, sint32 disp_range, sint32 pixels_in_pitch)
{
	const sint32 x_out_pitch = x / pixels_in_pitch;
	sint32 x_in_pitch = x - __mul24(pixels_in_pitch, x_out_pitch);
	return static_cast<uint8*>(cost.ptr) + __mul24(__mul24(y, cost.ysize), cost.pitch) + __mul24(x_out_pitch, cost.pitch) + __mul24(x_in_pitch, disp_range);
}


//reduction
template<class T>
__device__ void ReduceMin(T* sdata, uint32& min)		
{
#if 1
	sint32 tid = threadIdx.x;
	volatile T* smem = sdata;
	sint32 local_min = smem[tid];
	if (blockDim.x >= 256) {
		if (tid < 128) { smem[tid] = cu_min(smem[tid], smem[tid + 128]); } __syncthreads();
		if (tid < 64) { smem[tid] = cu_min(smem[tid], smem[tid + 64]); } __syncthreads();
		if (tid < 32) {
			smem[tid] = cu_min(smem[tid], smem[tid + 32]);
			smem[tid] = cu_min(smem[tid], smem[tid + 16]);
			smem[tid] = cu_min(smem[tid], smem[tid + 8]);
			smem[tid] = cu_min(smem[tid], smem[tid + 4]);
			smem[tid] = cu_min(smem[tid], smem[tid + 2]);
			smem[tid] = cu_min(smem[tid], smem[tid + 1]);
		}
	}
	if (blockDim.x >= 128) {
		if (tid < 64) { smem[tid] = cu_min(smem[tid], smem[tid + 64]); } __syncthreads();
		if (tid < 32) {
			smem[tid] = cu_min(smem[tid], smem[tid + 32]);
			smem[tid] = cu_min(smem[tid], smem[tid + 16]);
			smem[tid] = cu_min(smem[tid], smem[tid + 8]);
			smem[tid] = cu_min(smem[tid], smem[tid + 4]);
			smem[tid] = cu_min(smem[tid], smem[tid + 2]);
			smem[tid] = cu_min(smem[tid], smem[tid + 1]);
		}
	}
	else if (blockDim.x >= 64 && tid < 32) {
		local_min = cu_min(local_min, smem[tid + 32]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 16]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 8]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 4]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 2]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 1]);
		smem[tid] = local_min;
	}
	else if (blockDim.x >= 32 && tid < 16) {
		local_min = cu_min(local_min, smem[tid + 16]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 8]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 4]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 2]);
		smem[tid] = local_min;
		local_min = cu_min(local_min, smem[tid + 1]);
		smem[tid] = local_min;
	}
	__syncthreads();
	min = smem[0];
#else
	volatile T* smem = sdata;
	uint32 blockSize = blockDim.x;
	uint32 tid = threadIdx.x;
	if (blockSize >= 512) {
		if (tid < 256) { smem[tid] = cu_min(smem[tid] ,smem[tid + 256]); } __syncthreads();
	}
	if (blockSize >= 256) {
		if (tid < 128) { smem[tid] = cu_min(smem[tid], smem[tid + 128]); } __syncthreads();
	}
	if (blockSize >= 128) {
		if (tid < 64) { smem[tid] = cu_min(smem[tid], smem[tid + 64]); } __syncthreads();
	}
	if (tid < 32) {
		if (blockSize >= 64) smem[tid] = cu_min(smem[tid], smem[tid + 32]);
		if (blockSize >= 32) smem[tid] = cu_min(smem[tid], smem[tid + 16]);
		if (blockSize >= 16) smem[tid] = cu_min(smem[tid], smem[tid + 8]);
		if (blockSize >= 8) smem[tid] = cu_min(smem[tid], smem[tid + 4]);
		if (blockSize >= 4) smem[tid] = cu_min(smem[tid], smem[tid + 2]);
		if (blockSize >= 2) smem[tid] = cu_min(smem[tid], smem[tid + 1]);
	}
	__syncthreads();
	min = smem[0];
#endif
}

//parabola
__device__ float32 SubPixBias(float32 c1, float32 c0, float32 c2)
{
	float32 denom = cu_max(1.0f, c1 + c2 - 2 * c0);
	return (c1 - c2 + denom) / (2.0f * denom);
}

__global__ void Kernel_Aggregate_Up2Down(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1,sint32 p2)
{
	//采用N（N为32的整数倍）线程计算同一个像素所有视差范围的策略
	//一个线程负责两个视差，提高规约效率，因此N=disp_range/2	
	//而规约算法要求N>=32，且N为32的整数倍，这就约束了disp_range，即disp_range>=64，且为64整数倍
	//其他路径均同样策略
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_u[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last = 0;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];
		
		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_u[data_offset + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_u[data_offset + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_u[data_offset + threadIdx.x + blockDim.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_u[data_offset + threadIdx.x + blockDim.x + 1] + p1;
		
		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);
		
		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_u[data_offset + threadIdx.x] = aggr_1;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_u[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Kernel_Aggregate_Up2Down(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//采用N（N为32的整数倍）线程计算同一个像素所有视差范围的策略
	//一个线程负责两个视差，提高规约效率，因此N=disp_range/2	
	//而规约算法要求N>=32，且N为32的整数倍，这就约束了disp_range，即disp_range>=64，且为64整数倍
	//其他路径均同样策略
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_u[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + yoffset * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 4;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;

		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yoffset + i + 1) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			continue;
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_u[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_u[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_u[data_offset + idx_2] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_u[data_offset + threadIdx.x] = aggr_1;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_u[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Kernel_Aggregate_Up2Down(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	//采用N（N为32的整数倍）线程计算同一个像素所有视差范围的策略
	//一个线程负责两个视差，提高规约效率，因此N=disp_range/2	
	//而规约算法要求N>=32，且N为32的整数倍，这就约束了disp_range，即disp_range>=64，且为64整数倍
	//其他路径均同样策略
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_u[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_u[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + yoffset * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	uint8* bytes = img_bytes + yoffset * im_psize + image_x;
	uint8 lastBytes = *bytes;
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 4;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		bytes += im_psize;

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yoffset + i + 1) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			lastBytes = base_bytes;
			continue;
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_u[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + cu_max(p1, p2 / (abs(base_bytes - lastBytes) + 1));

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_u[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_u[data_offset + idx_2] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_u[data_offset + threadIdx.x] = aggr_1;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_u[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}
__global__ void Aggregate_U_Kernel_1for1(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//一个线程负责一个视差
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_u[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = 0xFF;
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;


		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (i + 1) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_u[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_u[data_offset + threadIdx.x] = aggr;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr;

		__syncthreads();

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}

template <sint32 DIRECTION>
__global__ void Kernel_Aggregate_Vertical_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//只针对视差范围为32的情况
	//采用32线程计算同一个像素所有视差范围的策略
	//一个线程负责一个视差
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, DIRECTION == 0 ? yoffset : yend - 1, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_u[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + yoffset * idp_psize) + image_x);
	sint32 costptr_step = DIRECTION == 0 ? cost_init.ysize * cost_init.pitch : -cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, DIRECTION == 0 ? yoffset : yend - 1, disp_range, pixels_in_pitch);
	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;

		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (DIRECTION == 0 ? yoffset + i + 1 : yend - 2 - i) * idp_psize) + image_x);

		if (base_d == INVALID_VALUE_SHORT) {
			//last_base_d = base_d;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_u[data_offset + idx_1] + p1;

		Lr_4 = min_cost_last + cu_max(p1, p2 / abs(last_base_d - base_d + 1));

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_u[data_offset + threadIdx.x] = aggr;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x, min_cost_last);
		//min_cost_last = warpShflReduceMin(aggr);
		last_base_d = base_d;
	}
}
__global__ void Aggregate_U_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//只针对视差范围为32的情况
	//采用32线程计算同一个像素所有视差范围的策略
	//一个线程负责一个视差
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_u[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + yoffset * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);
	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;

		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yoffset + i + 1) * idp_psize) + image_x);

		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_u[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_u[data_offset + threadIdx.x] = aggr;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Aggregate_U_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	//只针对视差范围为32的情况
	//采用32线程计算同一个像素所有视差范围的策略
	//一个线程负责一个视差
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_u[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_u[threadIdx.y * blockDim.x * 3 + 2 * blockDim.x + threadIdx.x] = 0xFF;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + yoffset * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);
	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	uint8* bytes = img_bytes + yoffset * im_psize + image_x;
	uint8 lastBytes = *bytes;
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		bytes += im_psize;

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yoffset + i + 1) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			lastBytes = base_bytes;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_u[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_u[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + cu_max(p1, p2 / (abs(base_bytes - lastBytes) + 1));

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_u[data_offset + threadIdx.x] = aggr;
		sr_last_u[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}

__global__ void Kernel_Aggregate_Down2Up(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_d[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yend - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c1;
	sr_last_d[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c2;
	sr_last_d[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_d + threadIdx.y * blockDim.x * 3 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yend - 1, disp_range, pixels_in_pitch);
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x + blockDim.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x + blockDim.x + 1] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x] = aggr_1;
		sr_last_d[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_d[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_d + threadIdx.y * blockDim.x * 3 + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Kernel_Aggregate_Down2Up(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//采用N（N为32的整数倍）线程计算同一个像素所有视差范围的策略
	//一个线程负责两个视差，提高规约效率，因此N=disp_range/2	
	//而规约算法要求N>=32，且N为32的整数倍，这就约束了disp_range，即disp_range>=64，且为64整数倍
	//其他路径均同样策略
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_d[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yend - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_d[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_d[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_d[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_d + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 1) * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yend - 1, disp_range, pixels_in_pitch);

	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 4;
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;

		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 2 - i) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_d[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_d[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_d[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_d[data_offset + idx_2] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_d[data_offset + threadIdx.x] = aggr_1;
		sr_last_d[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_d[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_d + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Kernel_Aggregate_Down2Up(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	//采用N（N为32的整数倍）线程计算同一个像素所有视差范围的策略
	//一个线程负责两个视差，提高规约效率，因此N=disp_range/2	
	//而规约算法要求N>=32，且N为32的整数倍，这就约束了disp_range，即disp_range>=64，且为64整数倍
	//其他路径均同样策略
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_d[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yend - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_d[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_d[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_d[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_d + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 1) * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yend - 1, disp_range, pixels_in_pitch);

	//-------------------//

	//从下到上聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	uint8* bytes = img_bytes + (height - 1) * im_psize + image_x;
	uint8 lastBytes = *bytes;
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 4;
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		bytes -= im_psize;

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 2 - i) * idp_psize) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			lastBytes = base_bytes;
			continue;
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_d[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_d[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + cu_max(p1, p2 / (abs(base_bytes - lastBytes) + 1));

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_d[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_d[data_offset + idx_2] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_d[data_offset + threadIdx.x] = aggr_1;
		sr_last_d[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_d[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);

		__syncthreads();

		ReduceMin<uint32>(sr_last_d + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}
__global__ void Aggregate_D_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	//只针对视差范围为32的情况
	//采用32线程计算同一个像素所有视差范围的策略
	//一个线程负责一个视差
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_d[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, yend - 1, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_d[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_d[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_d + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 1) * idp_psize) + image_x);
	uint32 costptr_step = cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, yend - 1, disp_range, pixels_in_pitch);
	//-------------------//

	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;

		sint16 base_d = *((sint16*)((uint8*)init_disp_mat + (yend - 2 - i) * idp_psize) + image_x);

		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_d[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_d[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_d[data_offset + threadIdx.x] = aggr;
		sr_last_d[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_d + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}


__global__ void Kernel_Aggregate_Left2Right(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_l[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_l[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);


	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint32 costptr_step = disp_range;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_l[data_offset + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_l[data_offset + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_l[data_offset + threadIdx.x + blockDim.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_l[data_offset + threadIdx.x + blockDim.x + 1] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_l[data_offset + threadIdx.x] = aggr_1;
		sr_last_l[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_l[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_l + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Kernel_Aggregate_Left2Right(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_l[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_l[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);


	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xoffset);
	sint32 costptr_step = cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + i + xoffset + 1);
		if (base_d == INVALID_VALUE_SHORT)
			continue;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_l[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_l[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_l[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_l[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_l[data_offset + threadIdx.x] = aggr_1;
		sr_last_l[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_l[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_l + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Kernel_Aggregate_Left2Right(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_l[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_l[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_l[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);


	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xoffset);
	sint32 costptr_step = cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	uint8* bytes = img_bytes + image_y * im_psize + xoffset;
	uint8 lastBytes = *bytes;
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		bytes++;

		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + i + xoffset + 1);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			lastBytes = base_bytes;
			continue;
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_l[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_l[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + cu_max(p1, p2 / (abs(base_bytes - lastBytes) + 1));

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_l[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_l[data_offset + idx_2] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_l[data_offset + threadIdx.x] = aggr_1;
		sr_last_l[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_l[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_l + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}

template <sint32 DIRECTION>
__global__ void Kernel_Aggregate_Horizontal_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;


	extern __shared__ uint32 sr_last_l[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, DIRECTION == 0 ? xoffset : xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
#if 1
	sr_last_l[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_l[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
#else
	sr_last_l[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
#endif

	uint32 min_cost_last;

#if 1
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);
#else
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 3, min_cost_last);
#endif

	uint32 aggr = c;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + (DIRECTION == 0 ? xoffset : xEnd - 1));
	sint32 costptr_step = DIRECTION == 0 ? disp_range : -disp_range;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, DIRECTION == 0 ? xoffset : xEnd - 1, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//水平方向聚合，一个线程束负责32个视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + (DIRECTION == 0 ? i + xoffset + 1 : xEnd - 2 - i));
		if (base_d == INVALID_VALUE_SHORT) {
			//last_base_d = base_d;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
#if 1
			Lr_2 = sr_last_l[data_offset + idx_1] + p1;
#else
			Lr_2 = __shfl(aggr, idx_1, warpSize) + p1;
#endif
		sint32 idx_2 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
#if 1
			Lr_3 = sr_last_l[data_offset + idx_2] + p1;
#else
			Lr_3 = __shfl(aggr, idx_2, warpSize) + p1;
#endif

		Lr_4 = min_cost_last + cu_max(p1, p2 / abs(last_base_d - base_d + 1));

		aggr = cost + (cu_min(cu_min(aggr, Lr_2), cu_min(Lr_3, Lr_4)) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

#if 1
		sr_last_l[data_offset + threadIdx.x] = aggr;
		sr_last_l[data_offset + blockDim.x + threadIdx.x] = aggr;
		ReduceMin<uint32>(sr_last_l + data_offset + blockDim.x, min_cost_last);

#else
		sr_last_l[data_offset + threadIdx.x] = aggr;
		ReduceMin<uint32>(sr_last_l + data_offset, min_cost_last);
#endif

		last_base_d = base_d;
	}
}

template <sint32 DIRECTION>
__global__ void Kernel_Aggregate_Horizontal_Warpfor128(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_l[];

	sint32 data_offset = threadIdx.y * blockDim.x * 6;
	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, DIRECTION == 0 ? xoffset : xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	uint32 c3 = cost_base_src[threadIdx.x + 2 * blockDim.x];
	uint32 c4 = cost_base_src[threadIdx.x + 3 * blockDim.x];
	sr_last_l[data_offset + threadIdx.x] = c1;
	sr_last_l[data_offset + blockDim.x * 1 + threadIdx.x] = c2;
	sr_last_l[data_offset + blockDim.x * 2 + threadIdx.x] = c3;
	sr_last_l[data_offset + blockDim.x * 3 + threadIdx.x] = c4;
	sr_last_l[data_offset + blockDim.x * 4 + threadIdx.x] = cu_min(cu_min(c1, c2), cu_min(c3, c4));
	sr_last_l[data_offset + blockDim.x * 5 + threadIdx.x] = 0xFF;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_l + threadIdx.y * blockDim.x * 6 + blockDim.x * 4, min_cost_last);

	uint32 aggr[4];
	aggr[0] = c1;
	aggr[1] = c2;
	aggr[2] = c3;
	aggr[3] = c4;

	sint32 costptr_step = DIRECTION == 0 ? disp_range : -disp_range;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, DIRECTION == 0 ? xoffset : xEnd - 1, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责4*N个候选视差计算，1个线程负责4个候选视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++)	{
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;

		uint32 Lr_2[4], Lr_3[4], Lr_4;
		uint32 cost[4];
		uint32 d[4];
		Lr_4 = min_cost_last + p2;

		cost[0] = cost_base_src[threadIdx.x];
		cost[1] = cost_base_src[threadIdx.x + 1 * blockDim.x];
		cost[2] = cost_base_src[threadIdx.x + 2 * blockDim.x];
		cost[3] = cost_base_src[threadIdx.x + 3 * blockDim.x];

		d[0] = threadIdx.x;
		d[1] = threadIdx.x + 1 * blockDim.x;
		d[2] = threadIdx.x + 2 * blockDim.x;
		d[3] = threadIdx.x + 3 * blockDim.x;

		if (d[0] == 0)
			Lr_2[0] = INVALID_COST;
		else
			Lr_2[0] = sr_last_l[data_offset + d[0] - 1] + p1;
		if (d[0] == disp_range - 1)
			Lr_3[0] = INVALID_COST;
		else
			Lr_3[0] = sr_last_l[data_offset + d[0] + 1] + p1;
		if (d[1] == 0)
			Lr_2[1] = INVALID_COST;
		else
			Lr_2[1] = sr_last_l[data_offset + d[1] - 1] + p1;
		if (d[1] == disp_range - 1)
			Lr_3[1] = INVALID_COST;
		else
			Lr_3[1] = sr_last_l[data_offset + d[1] + 1] + p1;
		if (d[2] == 0)
			Lr_2[2] = INVALID_COST;
		else
			Lr_2[2] = sr_last_l[data_offset + d[2] - 1] + p1;
		if (d[2] == disp_range - 1)
			Lr_3[2] = INVALID_COST;
		else
			Lr_3[2] = sr_last_l[data_offset + d[2] + 1] + p1;
		if (d[3] == 0)
			Lr_2[3] = INVALID_COST;
		else
			Lr_2[3] = sr_last_l[data_offset + d[3] - 1] + p1;
		if (d[3] == disp_range - 1)
			Lr_3[3] = INVALID_COST;
		else
			Lr_3[3] = sr_last_l[data_offset + d[3] + 1] + p1;

		aggr[0] = cost[0] + (cu_min(cu_min(cu_min(aggr[0], Lr_2[0]), Lr_3[0]), Lr_4) - min_cost_last);
		aggr[1] = cost[1] + (cu_min(cu_min(cu_min(aggr[1], Lr_2[1]), Lr_3[1]), Lr_4) - min_cost_last);
		aggr[2] = cost[2] + (cu_min(cu_min(cu_min(aggr[2], Lr_2[2]), Lr_3[2]), Lr_4) - min_cost_last);
		aggr[3] = cost[3] + (cu_min(cu_min(cu_min(aggr[3], Lr_2[3]), Lr_3[3]), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr[0];
		cost_base_dst[threadIdx.x + 1 * blockDim.x] = aggr[1];
		cost_base_dst[threadIdx.x + 2 * blockDim.x] = aggr[2];
		cost_base_dst[threadIdx.x + 3 * blockDim.x] = aggr[3];

		sr_last_l[data_offset + threadIdx.x] = aggr[0];
		sr_last_l[data_offset + threadIdx.x + 1 * blockDim.x] = aggr[1];
		sr_last_l[data_offset + threadIdx.x + 2 * blockDim.x] = aggr[2];
		sr_last_l[data_offset + threadIdx.x + 3 * blockDim.x] = aggr[3];

		sr_last_l[data_offset + blockDim.x * 4 + threadIdx.x] = cu_min(cu_min(aggr[0], aggr[1]), cu_min(aggr[2], aggr[3]));

		ReduceMin<uint32>(sr_last_l + data_offset + blockDim.x * 4, min_cost_last);
	}
}
template <sint32 DIRECTION>
__global__ void Kernel_Aggregate_Vertical_Warpfor128(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 yend, sint32 p1, sint32 p2)
{
	//采用32线程计算同一个像素128视差范围的策略
	//一个线程负责4个视差，提高规约效率
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_u[];

	sint32 data_offset = threadIdx.y * blockDim.x * 6;
	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = GetCostPtr(cost_init, image_x, DIRECTION == 0 ? yoffset : yend - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	uint32 c3 = cost_base_src[threadIdx.x + 2 * blockDim.x];
	uint32 c4 = cost_base_src[threadIdx.x + 3 * blockDim.x];
	sr_last_u[data_offset + threadIdx.x] = c1;
	sr_last_u[data_offset + blockDim.x * 1 + threadIdx.x] = c2;
	sr_last_u[data_offset + blockDim.x * 2 + threadIdx.x] = c3;
	sr_last_u[data_offset + blockDim.x * 3 + threadIdx.x] = c4;
	sr_last_u[data_offset + blockDim.x * 4 + threadIdx.x] = cu_min(cu_min(c1, c2), cu_min(c3, c4));

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x * 4, min_cost_last);

	uint32 aggr[4];
	aggr[0] = c1;
	aggr[1] = c2;
	aggr[2] = c3;
	aggr[3] = c4;

	sint32 costptr_step = (DIRECTION == 0) ? cost_init.ysize * cost_init.pitch : -cost_init.ysize * cost_init.pitch;
	uint8* cost_base_dst = GetCostPtr(cost_path, image_x, (DIRECTION == 0) ? yoffset : yend - 1, disp_range, pixels_in_pitch);

	//-------------------//
	//从上到下聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < yend - 1 - yoffset; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;

		uint32 Lr_2[4], Lr_3[4], Lr_4;
		uint32 cost[4];
		uint32 d[4];
		Lr_4 = min_cost_last + p2;

		cost[0] = cost_base_src[threadIdx.x];
		cost[1] = cost_base_src[threadIdx.x + 1 * blockDim.x];
		cost[2] = cost_base_src[threadIdx.x + 2 * blockDim.x];
		cost[3] = cost_base_src[threadIdx.x + 3 * blockDim.x];

		d[0] = threadIdx.x;
		d[1] = threadIdx.x + 1 * blockDim.x;
		d[2] = threadIdx.x + 2 * blockDim.x;
		d[3] = threadIdx.x + 3 * blockDim.x;

		if (d[0] == 0)
			Lr_2[0] = INVALID_COST;
		else
			Lr_2[0] = sr_last_u[data_offset + d[0] - 1] + p1;
		if (d[0] == disp_range - 1)
			Lr_3[0] = INVALID_COST;
		else
			Lr_3[0] = sr_last_u[data_offset + d[0] + 1] + p1;
		if (d[1] == 0)
			Lr_2[1] = INVALID_COST;
		else
			Lr_2[1] = sr_last_u[data_offset + d[1] - 1] + p1;
		if (d[1] == disp_range - 1)
			Lr_3[1] = INVALID_COST;
		else
			Lr_3[1] = sr_last_u[data_offset + d[1] + 1] + p1;
		if (d[2] == 0)
			Lr_2[2] = INVALID_COST;
		else
			Lr_2[2] = sr_last_u[data_offset + d[2] - 1] + p1;
		if (d[2] == disp_range - 1)
			Lr_3[2] = INVALID_COST;
		else
			Lr_3[2] = sr_last_u[data_offset + d[2] + 1] + p1;
		if (d[3] == 0)
			Lr_2[3] = INVALID_COST;
		else
			Lr_2[3] = sr_last_u[data_offset + d[3] - 1] + p1;
		if (d[3] == disp_range - 1)
			Lr_3[3] = INVALID_COST;
		else
			Lr_3[3] = sr_last_u[data_offset + d[3] + 1] + p1;

		aggr[0] = cost[0] + (cu_min(cu_min(cu_min(aggr[0], Lr_2[0]), Lr_3[0]), Lr_4) - min_cost_last);
		aggr[1] = cost[1] + (cu_min(cu_min(cu_min(aggr[1], Lr_2[1]), Lr_3[1]), Lr_4) - min_cost_last);
		aggr[2] = cost[2] + (cu_min(cu_min(cu_min(aggr[2], Lr_2[2]), Lr_3[2]), Lr_4) - min_cost_last);
		aggr[3] = cost[3] + (cu_min(cu_min(cu_min(aggr[3], Lr_2[3]), Lr_3[3]), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr[0];
		cost_base_dst[threadIdx.x + 1 * blockDim.x] = aggr[1];
		cost_base_dst[threadIdx.x + 2 * blockDim.x] = aggr[2];
		cost_base_dst[threadIdx.x + 3 * blockDim.x] = aggr[3];

		sr_last_u[data_offset + threadIdx.x] = aggr[0];
		sr_last_u[data_offset + threadIdx.x + 1 * blockDim.x] = aggr[1];
		sr_last_u[data_offset + threadIdx.x + 2 * blockDim.x] = aggr[2];
		sr_last_u[data_offset + threadIdx.x + 3 * blockDim.x] = aggr[3];

		sr_last_u[data_offset + blockDim.x * 4 + threadIdx.x] = cu_min(cu_min(aggr[0], aggr[1]), cu_min(aggr[2], aggr[3]));

		ReduceMin<uint32>(sr_last_u + data_offset + blockDim.x * 4, min_cost_last);
	}
}


__global__ void Kernel_Aggregate_Right2Left(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_r[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_r[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c1;
	sr_last_r[threadIdx.y * blockDim.x * 3 + threadIdx.x + blockDim.x] = c2;
	sr_last_r[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_r + threadIdx.y * blockDim.x * 3 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint32 costptr_step = disp_range;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xEnd - 1, image_y, disp_range, pixels_in_pitch);
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_r[data_offset + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_r[threadIdx.y * blockDim.x + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_r[data_offset + threadIdx.x + blockDim.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_r[data_offset + threadIdx.x + blockDim.x + 1] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_r[data_offset + threadIdx.x] = aggr_1;
		sr_last_r[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_r[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_r + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Kernel_Aggregate_Right2Left(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_r[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_r[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_r[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_r[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_r + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);


	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 1);
	sint32 costptr_step = cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xEnd - 1, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 2 - i);
		if (base_d == INVALID_VALUE_SHORT)
			continue;

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_r[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_r[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_r[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_r[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_r[data_offset + threadIdx.x] = aggr_1;
		sr_last_r[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_r[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_r + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Kernel_Aggregate_Right2Left(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_r[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_r[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_r[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_r[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_r + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);


	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 1);
	sint32 costptr_step = cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xEnd - 1, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从左到右聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	uint8* bytes = img_bytes + image_y * im_psize + xEnd - 1;
	uint8 lastBytes = *bytes;
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		bytes--;

		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 2 - i);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			lastBytes = base_bytes;
			continue;
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_r[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_r[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + cu_max(p1, p2 / (abs(base_bytes - lastBytes) + 1));

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_r[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_r[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_r[data_offset + threadIdx.x] = aggr_1;
		sr_last_r[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_r[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_r + data_offset + blockDim.x * 2, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}
__global__ void Aggregate_R_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_y = blockIdx.x * blockDim.y + threadIdx.y + yoffset;
	if (image_y >= height)
		return;

	extern __shared__ uint32 sr_last_r[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, xEnd - 1, image_y, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_r[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_r[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_r + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);


	uint32 aggr = c;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 1);
	sint32 costptr_step = disp_range;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, xEnd - 1, image_y, disp_range, pixels_in_pitch);

	//-------------------//
	//从右到左聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	for (sint32 i = 0; i < xEnd - xoffset - 1; i++) {
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + image_y * idp_psize)) + xEnd - 2 - i);
		if (base_d == INVALID_VALUE_SHORT) {
			last_base_d = base_d;
			min_cost_last = INVALID_COST;
			continue;
		}

		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_r[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_r[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_r[data_offset + threadIdx.x] = aggr;
		sr_last_r[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_r + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}

__global__ void Aggregate_LU_Kernel(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_lu[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_lu[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_lu[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_lu[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_lu + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从左上到右下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x + i + 1 == xEnd) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, i + 1, disp_range, pixels_in_pitch);
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_lu[data_offset + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_lu[data_offset + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_lu[data_offset + blockDim.x + threadIdx.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_lu[data_offset + blockDim.x + threadIdx.x + 1] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_lu[data_offset + threadIdx.x] = aggr_1;
		sr_last_lu[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_lu[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_lu + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Aggregate_LU_Kernel_1for2(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_lu[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_lu[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_lu[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_lu[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_lu + threadIdx.y * blockDim.x * 3 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + yoffset * idp_psize)) + image_x);
	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从左上到右下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x++;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x == xEnd) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, i + 1, disp_range, pixels_in_pitch);
			image_x = xoffset;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		if (base_d == INVALID_VALUE_SHORT) {
			cost_base_dst[threadIdx.x] = cost_1;
			cost_base_dst[threadIdx.x + blockDim.x] = cost_2;
			sr_last_lu[data_offset + threadIdx.x] = cost_1;
			sr_last_lu[data_offset + blockDim.x + threadIdx.x] = cost_2;
			min_cost_last = INVALID_VALUE;
			last_base_d = base_d;
			continue;
		}

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_lu[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_lu[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_lu[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_lu[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_lu[data_offset + threadIdx.x] = aggr_1;
		sr_last_lu[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_lu[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_lu + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Aggregate_LU_Kernel_1for1(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_lu[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_lu[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = 0xFF;
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_lu + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从左上到右下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x++;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xEnd) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, i + 1, disp_range, pixels_in_pitch);
			image_x = xoffset;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_lu[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_lu[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		last_base_d = base_d;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_lu[data_offset + threadIdx.x] = aggr;
		sr_last_lu[data_offset + blockDim.x + threadIdx.x] = aggr;
		__syncthreads();

		ReduceMin<uint32>(sr_last_lu + data_offset + blockDim.x, min_cost_last);
	}
}
__global__ void Aggregate_LU_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_lu[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_lu[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + 2 * blockDim.x + threadIdx.x] = 0xFF;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_lu + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从左上到右下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x++;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xEnd) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, i + 1, disp_range, pixels_in_pitch);
			image_x = xoffset;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_lu[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_lu[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		last_base_d = base_d;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_lu[data_offset + threadIdx.x] = aggr;
		sr_last_lu[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_lu + data_offset + blockDim.x, min_cost_last);
	}
}
__global__ void Aggregate_LU_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_lu[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_lu[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_lu[threadIdx.y * blockDim.x * 3 + 2 * blockDim.x + threadIdx.x] = 0xFF;
	//__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_lu + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从左上到右下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	uint8* bytes = img_bytes + yoffset * im_psize + image_x;
	uint8 lastBytes = *bytes;
	bytes += im_psize + 1;
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x++;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xEnd) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, i + 1, disp_range, pixels_in_pitch);
			image_x = xoffset;
			bytes = img_bytes + (i + 1) * im_psize + image_x;
		}

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		bytes += im_psize + 1;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_lu[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_lu[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2 / (abs(base_bytes - lastBytes) + 1);

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_lu[data_offset + threadIdx.x] = aggr;
		sr_last_lu[data_offset + blockDim.x + threadIdx.x] = aggr;
		//__syncthreads();

		ReduceMin<uint32>(sr_last_lu + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}

__global__ void Aggregate_RD_Kernel_1for2(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 xEnd, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_rd[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, height - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_rd[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_rd[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_rd[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_rd + threadIdx.y * blockDim.x * 3 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + (height - 1) * idp_psize)) + image_x);
	sint32 costptr_step = cost_init.ysize * cost_init.pitch + cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, height - 1, disp_range, pixels_in_pitch);

	//-------------------//
	//右下到左上沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x--;
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x == xoffset) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xEnd - 1, height - 2 - i, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xEnd - 1, height - 2 - i, disp_range, pixels_in_pitch);
			image_x = xEnd - 1;
		}
		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (height - 2 - i) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			cost_base_dst[threadIdx.x] = cost_1;
			cost_base_dst[threadIdx.x + blockDim.x] = cost_2;
			sr_last_rd[data_offset + threadIdx.x] = cost_1;
			sr_last_rd[data_offset + blockDim.x + threadIdx.x] = cost_2;
			min_cost_last = INVALID_VALUE;
			last_base_d = base_d;
			continue;
		}

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_rd[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_rd[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_rd[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_rd[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_rd[data_offset + threadIdx.x] = aggr_1;
		sr_last_rd[data_offset + blockDim.x + threadIdx.x] = aggr_2;
		sr_last_rd[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_rd + data_offset + blockDim.x * 2, min_cost_last);
	}
}

__global__ void Aggregate_RU_Kernel(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ru[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_ru[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_ru[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_ru[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ru + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从右上到左下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x - i - 1 == xoffset - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, width - 1, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, width - 1, i + 1, disp_range, pixels_in_pitch);
		}

		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		if (threadIdx.x == 0)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_ru[data_offset + threadIdx.x - 1] + p1;
		Lr_3_1 = sr_last_ru[data_offset + threadIdx.x + 1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		Lr_2_2 = sr_last_ru[data_offset + blockDim.x + threadIdx.x - 1] + p1;
		if (threadIdx.x + blockDim.x == disp_range - 1)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_ru[data_offset + blockDim.x + threadIdx.x + 1] + p1;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_ru[data_offset + threadIdx.x] = aggr_1;
		sr_last_ru[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_ru[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_ru + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Aggregate_RU_Kernel_1for2(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ru[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_ru[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_ru[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_ru[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ru + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + yoffset * idp_psize)) + image_x);
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从右上到左下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x--;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x == xoffset - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, width - 1, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, width - 1, i + 1, disp_range, pixels_in_pitch);
			image_x = width - 1;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];
		if (base_d == INVALID_VALUE_SHORT) {
			cost_base_dst[threadIdx.x] = cost_1;
			cost_base_dst[threadIdx.x + blockDim.x] = cost_2;
			sr_last_ru[data_offset + threadIdx.x] = cost_1;
			sr_last_ru[data_offset + blockDim.x + threadIdx.x] = cost_2;
			min_cost_last = INVALID_VALUE;
			last_base_d = base_d;
			continue;
		}
		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_ru[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_ru[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_ru[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_ru[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_ru[data_offset + threadIdx.x] = aggr_1;
		sr_last_ru[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_ru[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_ru + data_offset + blockDim.x * 2, min_cost_last);
	}
}
__global__ void Aggregate_RU_Kernel_1for1(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ru[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_ru[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + blockDim.x * 2 + threadIdx.x] = 0xFF;
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ru + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从右上到左下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x--;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xoffset - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, width - 1, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, width - 1, i + 1, disp_range, pixels_in_pitch);
			image_x = width - 1;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_ru[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_ru[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_ru[data_offset + threadIdx.x] = aggr;
		sr_last_ru[data_offset + blockDim.x + threadIdx.x] = aggr;
		__syncthreads();

		ReduceMin<uint32>(sr_last_ru + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Aggregate_RU_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ru[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_ru[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + 2 * blockDim.x + threadIdx.x] = 0xFF;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ru + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;

	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从右上到左下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x--;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xoffset - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, width - 1, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, width - 1, i + 1, disp_range, pixels_in_pitch);
			image_x = width - 1;
		}
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_ru[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_ru[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_ru[data_offset + threadIdx.x] = aggr;
		sr_last_ru[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_ru + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
	}
}
__global__ void Aggregate_RU_Kernel_Warp(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range, 
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize, uint8* img_bytes, size_t im_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ru[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, yoffset, disp_range, pixels_in_pitch);
	uint32 c = cost_base_src[threadIdx.x];
	sr_last_ru[threadIdx.y * blockDim.x * 3 + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + blockDim.x + threadIdx.x] = c;
	sr_last_ru[threadIdx.y * blockDim.x * 3 + 2 * blockDim.x + threadIdx.x] = 0xFF;

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ru + threadIdx.y * blockDim.x * 3 + blockDim.x, min_cost_last);

	uint32 aggr = c;
	sint16 last_base_d = init_disp_mat[yoffset * idp_psize + image_x];
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, yoffset, disp_range, pixels_in_pitch);

	//-------------------//
	//从右上到左下沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	uint8* bytes = img_bytes + yoffset * im_psize + image_x;
	uint8 lastBytes = *bytes;
	bytes += im_psize - 1;
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x--;
		cost_base_src += costptr_step;
		cost_base_dst += costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		if (image_x == xoffset - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, width - 1, i + 1, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, width - 1, i + 1, disp_range, pixels_in_pitch);
			image_x = width - 1;
			bytes = img_bytes + (i + 1) * im_psize + image_x;
		}

		uint8 base_bytes = *bytes;
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (i + 1) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT)
			continue;
		bytes += im_psize - 1;
		uint32 cost = cost_base_src[threadIdx.x];

		uint32 Lr_2, Lr_3, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2 = INVALID_COST;
		else
			Lr_2 = sr_last_ru[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3 = INVALID_COST;
		else
			Lr_3 = sr_last_ru[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2 / (abs(base_bytes - lastBytes) + 1);

		aggr = cost + (cu_min(cu_min(cu_min(aggr, Lr_2), Lr_3), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr;

		sr_last_ru[data_offset + threadIdx.x] = aggr;
		sr_last_ru[data_offset + blockDim.x + threadIdx.x] = aggr;

		ReduceMin<uint32>(sr_last_ru + data_offset + blockDim.x, min_cost_last);
		last_base_d = base_d;
		lastBytes = base_bytes;
	}
}

__global__ void Aggregate_LD_Kernel_1for2(cudaPitchedPtr cost_init, cudaPitchedPtr cost_path, sint32 width, sint32 height, sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset, sint32 p1, sint32 p2, sint16* init_disp_mat, size_t idp_psize)
{
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	if (image_x >= width)
		return;

	extern __shared__ uint32 sr_last_ld[];

	sint32 pixels_in_pitch = cost_init.pitch / disp_range;
	uint8* cost_base_src = (uint8*)GetCostPtr(cost_init, image_x, height - 1, disp_range, pixels_in_pitch);
	uint32 c1 = cost_base_src[threadIdx.x];
	uint32 c2 = cost_base_src[threadIdx.x + blockDim.x];
	sr_last_ld[threadIdx.y * blockDim.x * 4 + threadIdx.x] = c1;
	sr_last_ld[threadIdx.y * blockDim.x * 4 + blockDim.x + threadIdx.x] = c2;
	sr_last_ld[threadIdx.y * blockDim.x * 4 + blockDim.x * 2 + threadIdx.x] = cu_min(c1, c2);
	__syncthreads();

	uint32 min_cost_last;
	ReduceMin<uint32>(sr_last_ld + threadIdx.y * blockDim.x * 4 + blockDim.x * 2, min_cost_last);

	uint32 aggr_1 = c1;
	uint32 aggr_2 = c2;

	sint16 last_base_d = *((sint16*)(((uint8*)init_disp_mat + (height - 1) * idp_psize)) + image_x);
	sint32 costptr_step = cost_init.ysize * cost_init.pitch - cost_init.pitch;
	uint8* cost_base_dst = (uint8*)GetCostPtr(cost_path, image_x, height - 1, disp_range, pixels_in_pitch);

	//-------------------//
	//从左下到右上沿对角线聚合，一个block内N个线程负责2*N个候选视差计算，1个线程负责两个候选视差
	//若碰到边界，则列号回到起始列号，行号递增规则不变
	for (sint32 i = 0; i < height - 1 - yoffset; i++) {
		image_x++;
		cost_base_src -= costptr_step;
		cost_base_dst -= costptr_step;
		sint32 data_offset = threadIdx.y * blockDim.x * 4;

		if (image_x == width - 1) {
			cost_base_src = (uint8*)GetCostPtr(cost_init, xoffset, height - 2 - i, disp_range, pixels_in_pitch);
			cost_base_dst = (uint8*)GetCostPtr(cost_path, xoffset, height - 2 - i, disp_range, pixels_in_pitch);
			image_x = xoffset;
		}
		uint32 cost_1 = cost_base_src[threadIdx.x];
		uint32 cost_2 = cost_base_src[threadIdx.x + blockDim.x];
		sint16 base_d = *((sint16*)(((uint8*)init_disp_mat + (height - 2 - i) * idp_psize)) + image_x);
		if (base_d == INVALID_VALUE_SHORT) {
			cost_base_dst[threadIdx.x] = cost_1;
			cost_base_dst[threadIdx.x + blockDim.x] = cost_2;
			sr_last_ld[data_offset + threadIdx.x] = cost_1;
			sr_last_ld[data_offset + blockDim.x + threadIdx.x] = cost_2;
			min_cost_last = INVALID_VALUE;
			last_base_d = base_d;
			continue;
		}

		uint32 Lr_2_1, Lr_3_1, Lr_4;
		sint32 idx_1 = threadIdx.x + base_d - 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_2_1 = INVALID_COST;
		else
			Lr_2_1 = sr_last_ld[data_offset + idx_1] + p1;
		idx_1 = threadIdx.x + base_d + 1 - last_base_d;
		if (idx_1 < 0 || idx_1 >= LEVEL_RANGE)
			Lr_3_1 = INVALID_COST;
		else
			Lr_3_1 = sr_last_ld[data_offset + idx_1] + p1;
		Lr_4 = min_cost_last + p2;

		uint32 Lr_2_2, Lr_3_2;
		sint32 idx_2 = threadIdx.x + blockDim.x + base_d - 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_2_2 = INVALID_COST;
		else
			Lr_2_2 = sr_last_ld[data_offset + idx_2] + p1;
		idx_2 = threadIdx.x + blockDim.x + base_d + 1 - last_base_d;
		if (idx_2 < 0 || idx_2 >= LEVEL_RANGE)
			Lr_3_2 = INVALID_COST;
		else
			Lr_3_2 = sr_last_ld[data_offset + idx_2] + p1;
		last_base_d = base_d;

		aggr_1 = cost_1 + (cu_min(cu_min(cu_min(aggr_1, Lr_2_1), Lr_3_1), Lr_4) - min_cost_last);
		aggr_2 = cost_2 + (cu_min(cu_min(cu_min(aggr_2, Lr_2_2), Lr_3_2), Lr_4) - min_cost_last);

		cost_base_dst[threadIdx.x] = aggr_1;
		cost_base_dst[threadIdx.x + blockDim.x] = aggr_2;

		sr_last_ld[data_offset + threadIdx.x] = aggr_1;
		sr_last_ld[data_offset + threadIdx.x + blockDim.x] = aggr_2;
		sr_last_ld[data_offset + blockDim.x * 2 + threadIdx.x] = cu_min(aggr_1, aggr_2);
		__syncthreads();

		ReduceMin<uint32>(sr_last_ld + data_offset + blockDim.x * 2, min_cost_last);
	}
}

__global__ void Kernel_ComputeDisparityByWTA(
	cudaPitchedPtr cost_path1, cudaPitchedPtr cost_path2, cudaPitchedPtr cost_path3, cudaPitchedPtr cost_path4,
	float32* disp_map, sint32 dp_psize,
	sint32 width, sint32 height,
	sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset,
	float32 uniqueness,
	sint16* init_disp_mat = nullptr, size_t idp_psize = 0)
{
	//将方向视差累加步骤和最佳视差计算放在同一个Kernel，减少全局内存的读取次数，充分利用共享内存
	//线程数为视差范围的一半，提高规约效率

	//子像素优化也放在此步骤，因为新开Kernel优化不仅要重新载入Cost数组，而且导致此Kernel一定要保存Cost数组，额外的增加了一存一读的耗时。放在此步骤可以直接利用共享内存，而且不用将最终的Cost数组存入
	//而且，由于Cost数组是以视差为变化最快维度排列的（即同一个像素的所有视差是连续的），所以新开Kernel无法达到线程合并访问，将大大影像效率
	//因此，放在此步骤，虽然线程利用率很低，（负责每个像素的所有线程中只有第一个线程参与子像素优化计算），但是应该是目前能够采取的最优方案了

	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	sint32 image_y = yoffset + blockIdx.y;
	const sint16 base_d = (init_disp_mat == nullptr) ? min_disparity_ : *((sint16*)((uint8*)(init_disp_mat)+image_y * idp_psize) + image_x);
	
	float32& disp = disp_map[image_y * dp_psize / sizeof(float32) + image_x];
	if (image_x > width - 5 || image_y > height - 5 || image_x < 5 || image_y < 5) {
		disp = INVALID_VALUE;
		return;
	}
	if (base_d == INVALID_VALUE_SHORT) {
		disp = INVALID_VALUE;
		return;
	}
	extern __shared__ uint32 sr_cost[];

	sint32 pixels_in_pitch = cost_path1.pitch / disp_range;
	uint8* cost_base_1 = GetCostPtr(cost_path1, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_2 = GetCostPtr(cost_path2, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_3 = GetCostPtr(cost_path3, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_4 = GetCostPtr(cost_path4, image_x, image_y, disp_range, pixels_in_pitch);

	const uint32 cost_path_1_1 = *(cost_base_1 + threadIdx.x);
	const uint32 cost_path_2_1 = *(cost_base_2 + threadIdx.x);
	const uint32 cost_path_3_1 = *(cost_base_3 + threadIdx.x);
	const uint32 cost_path_4_1 = *(cost_base_4 + threadIdx.x);

	const uint32 cost_path_1_2 = *(cost_base_1 + blockDim.x + threadIdx.x);
	const uint32 cost_path_2_2 = *(cost_base_2 + blockDim.x + threadIdx.x);
	const uint32 cost_path_3_2 = *(cost_base_3 + blockDim.x + threadIdx.x);
	const uint32 cost_path_4_2 = *(cost_base_4 + blockDim.x + threadIdx.x);

	uint32 cost1 = cost_path_1_1 + cost_path_2_1 + cost_path_3_1 + cost_path_4_1;
	uint32 cost2 = cost_path_1_2 + cost_path_2_2 + cost_path_3_2 + cost_path_4_2;

	sint32 data_offset = threadIdx.y * (4 * blockDim.x);
	sr_cost[data_offset + threadIdx.x] = cost1;
	sr_cost[data_offset + blockDim.x + threadIdx.x] = cost2;
	sr_cost[data_offset + 2 * blockDim.x + threadIdx.x] = cu_min(cost1, cost2);
	__syncthreads();

	uint32 minCost;
	ReduceMin<uint32>(sr_cost + data_offset + 2 * blockDim.x, minCost);
	if (cost1 == minCost) {
		sr_cost[data_offset + 2 * blockDim.x] = threadIdx.x;
	}
	else if (cost2 == minCost) {
		sr_cost[data_offset + 2 * blockDim.x] = threadIdx.x + blockDim.x;
	}
	__syncthreads();

	//确定整像素视差值，并计算子像素位置
	if (threadIdx.x == 0) {
		float32 unique_threshold = 1.0f - uniqueness;
		sint32 init_disp = sr_cost[data_offset + 2 * blockDim.x];
		if (init_disp <= 0 || init_disp >= disp_range - 1) {
			disp = INVALID_VALUE;
		}
		else {
			float32 c1 = sr_cost[data_offset + (init_disp - 1)];
			float32 c2 = minCost;
			float32 c3 = sr_cost[data_offset + (init_disp + 1)];
			float32 uniq = cu_min(c1 - c2, c3 - c2);
			if (uniq <= c2 * unique_threshold)
				disp = INVALID_VALUE;
			else {
				disp = init_disp + base_d + SubPixBias(c1, c2, c3);
			}
		}
	}
}

__global__ void Kernel_ComputeDisparityByWTA(
	cudaPitchedPtr cost_path1, cudaPitchedPtr cost_path2, cudaPitchedPtr cost_path3, cudaPitchedPtr cost_path4,
	cudaPitchedPtr cost_path5, cudaPitchedPtr cost_path6, cudaPitchedPtr cost_path7, cudaPitchedPtr cost_path8,
	float32* disp_map, sint32 dp_psize,
	sint32 width, sint32 height,
	sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset,
	float32 uniqueness,
	sint16* init_disp_mat, size_t idp_psize)
{
	//将方向视差累加步骤和最佳视差计算放在同一个Kernel，减少全局内存的读取次数，充分利用共享内存
	//线程数为视差范围的一半，提高规约效率

	//子像素优化也放在此步骤，因为新开Kernel优化不仅要重新载入Cost数组，而且导致此Kernel一定要保存Cost数组，额外的增加了一存一读的耗时。放在此步骤可以直接利用共享内存，而且不用将最终的Cost数组存入
	//而且，由于Cost数组是以视差为变化最快维度排列的（即同一个像素的所有视差是连续的），所以新开Kernel无法达到线程合并访问，将大大影像效率
	//因此，放在此步骤，虽然线程利用率很低，（负责每个像素的所有线程中只有第一个线程参与子像素优化计算），但是应该是目前能够采取的最优方案了

	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	sint32 image_y = yoffset + blockIdx.y;
	const sint16 base_d = (init_disp_mat == nullptr) ? min_disparity_ : *((sint16*)((uint8*)(init_disp_mat)+image_y * idp_psize) + image_x);
	float32& disp = disp_map[image_y * dp_psize / sizeof(float32) + image_x];
	if (image_x > width - 5 || image_y > height - 5 || image_x < 5 || image_y < 5) {
		disp = INVALID_VALUE;
		return;
	}
	if (base_d == INVALID_VALUE_SHORT) {
		disp = INVALID_VALUE;
		return;
	}
	extern __shared__ uint32 sr_cost[];

	sint32 pixels_in_pitch = cost_path1.pitch / disp_range;
	uint8* cost_base_1 = GetCostPtr(cost_path1, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_2 = GetCostPtr(cost_path2, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_3 = GetCostPtr(cost_path3, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_4 = GetCostPtr(cost_path4, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_5 = GetCostPtr(cost_path5, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_6 = GetCostPtr(cost_path6, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_7 = GetCostPtr(cost_path7, image_x, image_y, disp_range, pixels_in_pitch);
	uint8* cost_base_8 = GetCostPtr(cost_path8, image_x, image_y, disp_range, pixels_in_pitch);

	const uint32 cost_path_1_1 = *(cost_base_1 + threadIdx.x);
	const uint32 cost_path_2_1 = *(cost_base_2 + threadIdx.x);
	const uint32 cost_path_3_1 = *(cost_base_3 + threadIdx.x);
	const uint32 cost_path_4_1 = *(cost_base_4 + threadIdx.x);
	const uint32 cost_path_5_1 = *(cost_base_5 + threadIdx.x);
	const uint32 cost_path_6_1 = *(cost_base_6 + threadIdx.x);
	const uint32 cost_path_7_1 = *(cost_base_7 + threadIdx.x);
	const uint32 cost_path_8_1 = *(cost_base_8 + threadIdx.x);

	const uint32 cost_path_1_2 = *(cost_base_1 + blockDim.x + threadIdx.x);
	const uint32 cost_path_2_2 = *(cost_base_2 + blockDim.x + threadIdx.x);
	const uint32 cost_path_3_2 = *(cost_base_3 + blockDim.x + threadIdx.x);
	const uint32 cost_path_4_2 = *(cost_base_4 + blockDim.x + threadIdx.x);
	const uint32 cost_path_5_2 = *(cost_base_5 + blockDim.x + threadIdx.x);
	const uint32 cost_path_6_2 = *(cost_base_6 + blockDim.x + threadIdx.x);
	const uint32 cost_path_7_2 = *(cost_base_7 + blockDim.x + threadIdx.x);
	const uint32 cost_path_8_2 = *(cost_base_8 + blockDim.x + threadIdx.x);

	uint32 cost1 = cost_path_1_1 + cost_path_2_1 + cost_path_3_1 + cost_path_4_1 + cost_path_5_1 + cost_path_6_1 + cost_path_7_1 + cost_path_8_1;
	uint32 cost2 = cost_path_1_2 + cost_path_2_2 + cost_path_3_2 + cost_path_4_2 + cost_path_5_2 + cost_path_6_2 + cost_path_7_2 + cost_path_8_2;

	sint32 data_offset = threadIdx.y * (4 * blockDim.x);
	sr_cost[data_offset + threadIdx.x] = cost1;
	sr_cost[data_offset + blockDim.x + threadIdx.x] = cost2;
	sr_cost[data_offset + 2 * blockDim.x + threadIdx.x] = cu_min(cost1, cost2);
	__syncthreads();

	uint32 minCost;
	ReduceMin<uint32>(sr_cost + data_offset + 2 * blockDim.x, minCost);
	if (cost1 == minCost) {
		sr_cost[data_offset + 2 * blockDim.x] = threadIdx.x;
	}
	else if (cost2 == minCost) {
		sr_cost[data_offset + 2 * blockDim.x] = threadIdx.x + blockDim.x;
	}
	__syncthreads();

	//确定整像素视差值，并计算子像素位置
	if (threadIdx.x == 0) {
		float32 unique_threshold = 1.0f - uniqueness;
		sint32 init_disp = sr_cost[data_offset + 2 * blockDim.x];
		if (init_disp <= 0 || init_disp >= disp_range) {
			disp = INVALID_VALUE;
		}
		else {
			float32 c1 = sr_cost[data_offset + (init_disp - 1)];
			float32 c2 = minCost;
			float32 c3 = sr_cost[data_offset + (init_disp + 1)];
			float32 uniq = cu_min(c1 - c2, c3 - c2);
			if (uniq <= c2 * unique_threshold)
				disp = INVALID_VALUE;
			else {
				disp = init_disp + base_d + SubPixBias(c1, c2, c3);
			}
		}
	}
}

template <sint32 LINES>
__global__ void Kernel_ComputeDisparityByWTA_Warp(
	cudaPitchedPtr cost_path1, cudaPitchedPtr cost_path2, cudaPitchedPtr cost_path3, cudaPitchedPtr cost_path4,
	float32* disp_map, sint32 dp_psize,
	sint32 width, sint32 height,
	sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset,
	float32 uniqueness,
	sint16* init_disp_mat, size_t idp_psize,
	uint8* pBytes, sint32 im_psize)
{
	//将方向视差累加步骤和最佳视差计算放在同一个Kernel，减少全局内存的读取次数，充分利用共享内存
	//1个线程束负责32个视差
	//一个线程处理LINES行，增加指令集并行度

	//子像素优化也放在此步骤，因为新开Kernel优化不仅要重新载入Cost数组，而且导致此Kernel一定要保存Cost数组，额外的增加了一存一读的耗时。放在此步骤可以直接利用共享内存，而且不用将最终的Cost数组存入
	//而且，由于Cost数组是以视差为变化最快维度排列的（即同一个像素的所有视差是连续的），所以新开Kernel无法达到线程合并访问，将大大影像效率
	//因此，放在此步骤，虽然线程利用率很低，（负责每个像素的所有线程中只有第一个线程参与子像素优化计算），但是应该是目前能够采取的最优方案了
	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	sint32 image_y = yoffset + blockIdx.y * LINES;
	sint32 pixels_in_pitch = cost_path1.pitch / disp_range;
	sint32 data_offset = threadIdx.y * (3 * blockDim.x);
	
#pragma unroll 
	for (sint32 i = 0; i < LINES; i++, image_y++) {
		const sint16 base_d = *((sint16*)((uint8*)(init_disp_mat)+image_y * idp_psize) + image_x);
		float32& disp = *((float32*)((uint8*)(disp_map)+image_y * dp_psize) + image_x);
		if (image_x > width - 5 || image_y > height - 5 || image_x < 5 || image_y < 5) {
			disp = INVALID_VALUE;
			continue;
		}
		if (base_d == INVALID_VALUE_SHORT) {
			disp = INVALID_VALUE;
			continue;
		}
		extern __shared__ uint32 sr_cost[];

		uint8* cost_base_1 = GetCostPtr(cost_path1, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_2 = GetCostPtr(cost_path2, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_3 = GetCostPtr(cost_path3, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_4 = GetCostPtr(cost_path4, image_x, image_y, disp_range, pixels_in_pitch);

		const uint32 cost_path_1 = *(cost_base_1 + threadIdx.x);
		const uint32 cost_path_2 = *(cost_base_2 + threadIdx.x);
		const uint32 cost_path_3 = *(cost_base_3 + threadIdx.x);
		const uint32 cost_path_4 = *(cost_base_4 + threadIdx.x);

		uint32 cost = cost_path_1 + cost_path_2 + cost_path_3 + cost_path_4;

#if 0	//使用共享内存
		sr_cost[data_offset + threadIdx.x] = cost;
		sr_cost[data_offset + blockDim.x + threadIdx.x] = cost;
		sr_cost[data_offset + 2 * blockDim.x + threadIdx.x] = 0xFFFF;
#else	//使用共享内存+shfl
		sr_cost[data_offset + threadIdx.x] = cost;
#endif

		uint32 minCost;
		uint32 secMinCost;
#if 0
		ReduceMin<uint32>(sr_cost + data_offset + blockDim.x, minCost);
#else
		ReduceMin<uint32>(sr_cost + data_offset, minCost);

		uint32 cost1 = __shfl_up(cost, 1, warpSize);
		uint32 cost2 = __shfl_down(cost, 1, warpSize);

#endif
		if (cost == minCost) {
			//确定整像素视差值，并计算子像素位置
			float32 unique_threshold = 1.0f - uniqueness;
			sint32 init_disp = threadIdx.x;
			if (init_disp <= 1 || init_disp >= disp_range - 1) {
				disp = INVALID_VALUE;
			}
			else {
#if 0
				float32 c1 = sr_cost[data_offset + init_disp - 1];
				float32 c2 = minCost;
				float32 c3 = sr_cost[data_offset + init_disp + 1];
#else
				float32 c1 = static_cast<float32>(cost1);
				float32 c2 = minCost;
				float32 c3 = static_cast<float32>(cost2);

#endif
				float32 uniq = cu_min(c1 - c2, c3 - c2);
				if (uniq <= c2 * unique_threshold)
					disp = INVALID_VALUE;
				else {
					disp = init_disp + base_d + SubPixBias(c1, c2, c3);
				}
			}

		}
	}
}

template <sint32 LINES>
__global__ void Kernel_ComputeDisparityByWTA_Warpfor128(
	cudaPitchedPtr cost_path1, cudaPitchedPtr cost_path2, cudaPitchedPtr cost_path3, cudaPitchedPtr cost_path4,
	float32* disp_map, sint32 dp_psize,
	sint32 width, sint32 height,
	sint32 min_disparity_, sint32 disp_range,
	sint32 xoffset, sint32 yoffset,
	float32 uniqueness)
{
	//将方向视差累加步骤和最佳视差计算放在同一个Kernel，减少全局内存的读取次数，充分利用共享内存
	//1个线程束负责128个视差，提高规约效率、增加指令集并行度
	//一个线程处理LINES行，增加指令集并行度

	//子像素优化也放在此步骤，因为新开Kernel优化不仅要重新载入Cost数组，而且导致此Kernel一定要保存Cost数组，额外的增加了一存一读的耗时。放在此步骤可以直接利用共享内存，而且不用将最终的Cost数组存入
	//而且，由于Cost数组是以视差为变化最快维度排列的（即同一个像素的所有视差是连续的），所以新开Kernel无法达到线程合并访问，将大大影像效率
	//因此，放在此步骤，虽然线程利用率很低，（负责每个像素的所有线程中只有第一个线程参与子像素优化计算），但是应该是目前能够采取的最优方案了

	sint32 image_x = blockIdx.x * blockDim.y + threadIdx.y + xoffset;
	sint32 image_y = yoffset + blockIdx.y * LINES;
	sint32 pixels_in_pitch = cost_path1.pitch / disp_range;

#pragma unroll
	for (sint32 i = 0; i < LINES; i++, image_y++) {
		float32& disp = disp_map[image_y * dp_psize / sizeof(float32) + image_x];
		if (image_x > width - 5 || image_y > height - 5 || image_x < 5 || image_y < 5) {
			disp = INVALID_VALUE;
			continue;
		}

		extern __shared__ uint32 sr_cost[];

		uint8* cost_base_1 = GetCostPtr(cost_path1, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_2 = GetCostPtr(cost_path2, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_3 = GetCostPtr(cost_path3, image_x, image_y, disp_range, pixels_in_pitch);
		uint8* cost_base_4 = GetCostPtr(cost_path4, image_x, image_y, disp_range, pixels_in_pitch);

		const uint32 cost_path_1_1 = *(cost_base_1 + threadIdx.x);
		const uint32 cost_path_2_1 = *(cost_base_2 + threadIdx.x);
		const uint32 cost_path_3_1 = *(cost_base_3 + threadIdx.x);
		const uint32 cost_path_4_1 = *(cost_base_4 + threadIdx.x);
		 
		const uint32 cost_path_1_2 = *(cost_base_1 + blockDim.x + threadIdx.x);
		const uint32 cost_path_2_2 = *(cost_base_2 + blockDim.x + threadIdx.x);
		const uint32 cost_path_3_2 = *(cost_base_3 + blockDim.x + threadIdx.x);
		const uint32 cost_path_4_2 = *(cost_base_4 + blockDim.x + threadIdx.x);
		 
		const uint32 cost_path_1_3 = *(cost_base_1 + 2 * blockDim.x + threadIdx.x);
		const uint32 cost_path_2_3 = *(cost_base_2 + 2 * blockDim.x + threadIdx.x);
		const uint32 cost_path_3_3 = *(cost_base_3 + 2 * blockDim.x + threadIdx.x);
		const uint32 cost_path_4_3 = *(cost_base_4 + 2 * blockDim.x + threadIdx.x);
		 
		const uint32 cost_path_1_4 = *(cost_base_1 + 3 * blockDim.x + threadIdx.x);
		const uint32 cost_path_2_4 = *(cost_base_2 + 3 * blockDim.x + threadIdx.x);
		const uint32 cost_path_3_4 = *(cost_base_3 + 3 * blockDim.x + threadIdx.x);
		const uint32 cost_path_4_4 = *(cost_base_4 + 3 * blockDim.x + threadIdx.x);

		uint32 cost1 = cost_path_1_1 + cost_path_2_1 + cost_path_3_1 + cost_path_4_1;
		uint32 cost2 = cost_path_1_2 + cost_path_2_2 + cost_path_3_2 + cost_path_4_2;
		uint32 cost3 = cost_path_1_3 + cost_path_2_3 + cost_path_3_3 + cost_path_4_3;
		uint32 cost4 = cost_path_1_4 + cost_path_2_4 + cost_path_3_4 + cost_path_4_4;

		sint32 data_offset = threadIdx.y * (6 * blockDim.x);
		sr_cost[data_offset + threadIdx.x] = cost1;
		sr_cost[data_offset + 1 * blockDim.x + threadIdx.x] = cost2;
		sr_cost[data_offset + 2 * blockDim.x + threadIdx.x] = cost3;
		sr_cost[data_offset + 3 * blockDim.x + threadIdx.x] = cost4;
		sr_cost[data_offset + 4 * blockDim.x + threadIdx.x] = cu_min(cu_min(cost1, cost2), cu_min(cost3, cost4));

		uint32 minCost;
		ReduceMin<uint32>(sr_cost + data_offset + 4 * blockDim.x, minCost);
		
		if (cost1 == minCost) {
			sr_cost[data_offset + 4 * blockDim.x] = threadIdx.x;
		}
		else if (cost2 == minCost) {
			sr_cost[data_offset + 4 * blockDim.x] = threadIdx.x + blockDim.x;
		}
		else if (cost3 == minCost) {
			sr_cost[data_offset + 4 * blockDim.x] = threadIdx.x + 2 * blockDim.x;
		}
		else if (cost4 == minCost) {
			sr_cost[data_offset + 4 * blockDim.x] = threadIdx.x + 3 * blockDim.x;
		}
		__syncthreads();
		//确定整像素视差值，并计算子像素位置
		if (threadIdx.x == 0) {
			float32 unique_threshold = 1.0f - uniqueness;
			sint32 init_disp = sr_cost[data_offset + 4 * blockDim.x];
			if (init_disp <= 0 || init_disp >= disp_range) {
				disp = INVALID_VALUE;
			}
			else {
				//compute the extreme value of  parabola 
				float32 c1 = sr_cost[data_offset + (init_disp - 1)];
				float32 c2 = minCost;
				float32 c3 = sr_cost[data_offset + (init_disp + 1)];
				float32 uniq = cu_min(c1 - c2, c3 - c2);
				if (uniq <= c2 * unique_threshold)
					disp = INVALID_VALUE;
				else {
					disp = init_disp + min_disparity_ + SubPixBias(c1, c2, c3);
				}
			}
		}
	}
}

__device__ void swap(float32& a, float32& b) { float32 tmp = a; a = b; b = tmp; }

__global__ void Kernel_Median3x3(float32* disp_map, sint32 width, sint32 height, sint32 dp_psize)
{
	//利用快速中值计算法进行中值滤波
	//利用共享内存，减少全局内存的读取次数
	sint32 image_x = (blockIdx.x == 0) ? (blockIdx.x * blockDim.x + threadIdx.x) : (blockIdx.x * blockDim.x - 2 * blockIdx.x + threadIdx.x);
	sint32 image_y = (blockIdx.y == 0) ? (blockIdx.y * blockDim.y + threadIdx.y) : (blockIdx.y * blockDim.y - 2 * blockIdx.y + threadIdx.y);
	if (image_x > width - 1 || image_y > height - 1 || image_x < 1 || image_y < 1) {
		return;
	}

	float32* disp = (float32*)((uint8*)disp_map + image_y * dp_psize);
	extern __shared__ float32 sr_median_pix[];
	float32 center;
	sr_median_pix[threadIdx.y * blockDim.x + threadIdx.x] = center = disp[image_x];
	__syncthreads();

	if (threadIdx.x == 0 || threadIdx.x == blockDim.x - 1 || threadIdx.y == 0 || threadIdx.y == blockDim.y - 1)
		return;

	float32 a1, a2, a3, b1, b2, b3, c1, c2, c3;

	a1 = sr_median_pix[(threadIdx.y - 1) * blockDim.x + threadIdx.x - 1];
	a2 = sr_median_pix[(threadIdx.y - 1) * blockDim.x + threadIdx.x];
	a3 = sr_median_pix[(threadIdx.y - 1) * blockDim.x + threadIdx.x + 1];
	b1 = sr_median_pix[(threadIdx.y) * blockDim.x + threadIdx.x - 1];
	b2 = center;
	b3 = sr_median_pix[(threadIdx.y) * blockDim.x + threadIdx.x + 1];
	c1 = sr_median_pix[(threadIdx.y + 1) * blockDim.x + threadIdx.x - 1];
	c2 = sr_median_pix[(threadIdx.y + 1) * blockDim.x + threadIdx.x];
	c3 = sr_median_pix[(threadIdx.y + 1) * blockDim.x + threadIdx.x + 1];

	//列排序
	if (a1 > b1) swap(a1, b1);
	if (a1 > c1) swap(a1, c1);
	if (b1 > c1) swap(b1, c1);
	if (a2 > b2) swap(a2, b2);
	if (a2 > c2) swap(a2, c2);
	if (b2 > c2) swap(b2, c2);
	if (a3 > b3) swap(a3, b3);
	if (a3 > c3) swap(a3, c3);
	if (b3 > c3) swap(b3, c3);
	//行排序
	if (a1 > a2) swap(a1, a2);
	if (a1 > a3) swap(a1, a3);
	if (a2 > a3) swap(a2, a3);
	if (b1 > b2) swap(b1, b2);
	if (b1 > b3) swap(b1, b3);
	if (b2 > b3) swap(b2, b3);
	if (c1 > c2) swap(c1, c2);
	if (c1 > c3) swap(c1, c3);
	if (c2 > c3) swap(c2, c3);
	//对角线排序
	if (c1 > b2) swap(c1, b2);
	if (c1 > a3) swap(c1, a3);
	if (b2 > a3) swap(b2, a3);

	//赋值
	disp[image_x] = b2;
}

__global__ void Kernel_LRCheck(float32* pDispL, float32* pDispR, sint32 width, sint32 height, sint32 dp_psize, float32 threshold)
{
	//左右一致性检查
	float32* left = (float32*)((uint8*)pDispL + blockIdx.y * dp_psize);
	float32* right = (float32*)((uint8*)pDispR + blockIdx.y * dp_psize);
	sint32 image_xl = blockIdx.x * blockDim.x + threadIdx.x;
	if (image_xl > width)
		return;
	float32 disp_l = left[image_xl];
	if (disp_l == INVALID_VALUE)
	{
		return;
	}
	float32 image_xrf = image_xl - disp_l;
	sint32 image_xr = (image_xrf > 0) ? sint32(image_xrf + 0.5) : sint32(image_xrf - 0.5);
	if (image_xr < 0 || image_xr >= width)
	{
		left[image_xl] = INVALID_VALUE;
		return;
	}
	float32 disp_r = right[image_xr];
	if (fabs(disp_l - disp_r) > threshold)
		left[image_xl] = right[image_xr] = INVALID_VALUE;
	else
		left[image_xl] = (disp_l + disp_r) / 2.0f;
}

}

CostAggregator::CostAggregator(): width_(0), height_(0), img_left_(nullptr), img_right_(nullptr), im_psize_(0),
                                  disp_map_(nullptr), dp_psize_(0), cost_init_(), cost_aggr_(), cost_aggr_dir_{},
                                  min_disparity_(0), max_disparity_(0),
                                  ca_p1_(0), ca_p2_(0), constant_p2_(false), uniquess_(0),
                                  is_initialized_(false) { }

CostAggregator::~CostAggregator()
{
	
}

bool CostAggregator::Initialize(const sint32& width, const sint32& height, const sint32& min_disparity,
	const sint32& max_disparity)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	if (width_ <= 0 || height_ <= 0 || min_disparity_ >= max_disparity_) {
		is_initialized_ = false;
		return  false;
	}
	const sint32& disp_range = max_disparity_ - min_disparity_;

	// disparity map
	if (!CudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&disp_map_), &dp_psize_, size_t(width_) * sizeof(float32), size_t(height_)))||
		!CudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&disp_map_r_), &dp_psize_, size_t(width_) * sizeof(float32), size_t(height_)))) {
		return false;
	}

	// cost mat
	cudaExtent extent = make_cudaExtent(32, 32, 32);
	cudaPitchedPtr temp{};
	if (!CudaSafeCall(cudaMalloc3D(&temp, extent))) {
		is_initialized_ = false;
		return false;
	}

	// malloc aligned 3d array
	extent = make_cudaExtent(temp.pitch, size_t(width_) / (cu_max(1, temp.pitch / disp_range)), height_);
	cudaFree(temp.ptr);

	if (!CudaSafeCall(cudaMalloc3D(&cost_aggr_, extent))) {
		return false;
	}
	for (int k = 0; k < PATH_NUM; k++) {
		if (!CudaSafeCall(cudaMalloc3D(&cost_aggr_dir_[k], extent))) {
			return false;
		}
	}

	//crate streams
	cu_streams_ = static_cast<cudaStream_t*>(new cudaStream_t[8]);
	for (sint32 i = 0; i < 8; i++) {
		if (!CudaSafeCall(cudaStreamCreate(&(static_cast<cudaStream_t*>(cu_streams_))[i]))) {
			is_initialized_ = false;
			return false;
		}
	}

	is_initialized_ = true;
	return true;
}

void CostAggregator::SetData(uint8* img_left, uint8* img_right, const size_t& im_psize,
	cudaPitchedPtr* cost_init)
{
	img_left_ = img_left;
	img_right_ = img_right;
	im_psize_ = im_psize;
	cost_init_ = cost_init;
}

void CostAggregator::SetParam(const float32& p1, const float32& p2, const bool& constant_p2, const float32& uniquess)
{
	ca_p1_ = p1;
	ca_p2_ = p2;
	constant_p2_ = constant_p2;
	uniquess_ = uniquess;
}

void CostAggregator::Aggregate(sint16* init_disp_mat, const size_t& idp_psize, const StereoROI_T* ste_roi)
{
	if (!is_initialized_ || width_ <= 0 || height_ <= 0 || min_disparity_ >= max_disparity_ || !cost_init_) {
		return;
	}
	const sint32& disp_range = max_disparity_ - min_disparity_;

	sint32 roi_x = 0, roi_y = 0, roi_w = width_, roi_h = height_;
	if (ste_roi) {
		roi_x =ste_roi->x; roi_y =ste_roi->y;
		roi_w =ste_roi->w; roi_h =ste_roi->h;
	}
	else {
		roi_x = cu_max(0, min_disparity_); roi_y = 0;
		roi_w = width_ - roi_x; roi_h = height_;
	}

	auto& cost_init = *cost_init_;
	auto streams = static_cast<cudaStream_t*>(cu_streams_);
	dim3 threads_CA(disp_range / 2, cu_max(1, THREADS_COMMON * 2 / disp_range));
	dim3 block_CA(ceil((roi_w * 1.0) / threads_CA.y), 1);
	dim3 block2_CA(ceil((roi_h * 1.0) / threads_CA.y), 1);
	if (init_disp_mat && idp_psize > 0) {
		if (disp_range == 64) {
			if (constant_p2_) {
				cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Left2Right << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Right2Left << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				
			}
			else {
				cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_,
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize, img_left_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Left2Right << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize, img_left_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize, img_left_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Right2Left << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_,
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize, img_left_, im_psize_);
				
			}

			if (PATH_NUM == 8) {
				cusgm_ca::Aggregate_LU_Kernel_1for2 << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[4] >> > (cost_init, cost_aggr_dir_[4], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Aggregate_RD_Kernel_1for2 << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[7] >> > (cost_init, cost_aggr_dir_[5], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Aggregate_RU_Kernel_1for2 << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[5] >> > (cost_init, cost_aggr_dir_[6], width_, height_,
					min_disparity_, disp_range, roi_x, roi_y, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Aggregate_LD_Kernel_1for2 << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[6] >> > (cost_init, cost_aggr_dir_[7], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
			}
			for (sint32 i = 0; i < PATH_NUM; i++)
				cudaStreamSynchronize(streams[i]);

			//将各个方向代价相加到最终的代价数组中,然后统计最小代价
			dim3 threads_PA(disp_range / 2, THREADS_COMMON * 2 / disp_range);
			dim3 block_PA(ceil((roi_w * 1.0) / threads_PA.y), height_);
			if (PATH_NUM == 4)
				cusgm_ca::Kernel_ComputeDisparityByWTA << <block_PA, threads_PA, 4 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[2], cost_aggr_dir_[1], cost_aggr_dir_[3], 
					disp_map_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_, init_disp_mat, idp_psize);
			else if (PATH_NUM == 8)
				cusgm_ca::Kernel_ComputeDisparityByWTA << <block_PA, threads_PA, 4 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[2], cost_aggr_dir_[1], cost_aggr_dir_[3], 
					cost_aggr_dir_[4], cost_aggr_dir_[6], cost_aggr_dir_[7],cost_aggr_dir_[5], 
					disp_map_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_, init_disp_mat, idp_psize);
		}
		else if (disp_range == 32) {
			threads_CA.x = disp_range;
			threads_CA.y = THREADS_COMMON / disp_range;
			block_CA.x = ceil((roi_w * 1.0) / threads_CA.y);
			block2_CA.x = ceil((roi_h * 1.0) / threads_CA.y);
			if (constant_p2_) {
				cusgm_ca::Kernel_Aggregate_Vertical_Warp<0> << <block_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Horizontal_Warp<0> << <block2_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Vertical_Warp<1> << <block_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_,
					min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Horizontal_Warp<1> << <block2_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_, 
					min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_, init_disp_mat, idp_psize);
			}
			for (sint32 i = 0; i < 4; i++)
				cudaStreamSynchronize(streams[i]);
			//将各个方向代价相加到最终的代价数组中,然后统计最小代价
			const sint32 n_lines = 8;
			dim3 threads_PA(disp_range, THREADS_COMMON / disp_range);
			dim3 block_PA(ceil((roi_w * 1.0) / threads_PA.y), roi_h / n_lines);
			cusgm_ca::Kernel_ComputeDisparityByWTA_Warp<n_lines> << <block_PA, threads_PA, 3 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[2], cost_aggr_dir_[1], cost_aggr_dir_[3], 
				disp_map_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_, init_disp_mat, idp_psize, img_right_, im_psize_);
		}
	}
	else
	{
		if (disp_range != 128) {
			cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Left2Right << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Right2Left << <block2_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_);
		}
		else {
			dim3 threadsWarp(32, cu_max(1, THREADS_COMMON / 32));
			dim3 blockWarp(ceil((roi_w * 1.0) / threadsWarp.y), 1);
			dim3 blockWarp2(ceil((roi_h * 1.0) / threadsWarp.y), 1);
			cusgm_ca::Kernel_Aggregate_Vertical_Warpfor128<0> << <blockWarp, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Horizontal_Warpfor128<0> << <blockWarp2, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Vertical_Warpfor128<1> << <blockWarp, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_,ca_p2_);
			cusgm_ca::Kernel_Aggregate_Horizontal_Warpfor128<1> << <blockWarp2, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_, 
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_,ca_p2_);
		}
		for (sint32 i = 0; i < 4; i++)
			cudaStreamSynchronize(streams[i]);

		//将各个方向代价相加到最终的代价数组中,然后统计最小代价
		if (disp_range != 128) {
			dim3 threads_PA(disp_range / 2, cu_max(1, THREADS_COMMON * 2 / disp_range));
			dim3 block_PA(ceil((roi_w * 1.0) / threads_PA.y), roi_h);
			cusgm_ca::Kernel_ComputeDisparityByWTA << <block_PA, threads_PA, 4 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3],
				disp_map_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_);
		}
		else {
			const sint32 n_lines = 2;
			dim3 threadsWarp_PA(32, cu_max(1, THREADS_COMMON / 32));
			dim3 blockWarp_PA(ceil((roi_w * 1.0) / threadsWarp_PA.y), roi_h / n_lines);
			cusgm_ca::Kernel_ComputeDisparityByWTA_Warpfor128<n_lines> << <blockWarp_PA, threadsWarp_PA, 6 * threadsWarp_PA.x * threadsWarp_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3],
				disp_map_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_);
		}
	}
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}


void CostAggregator::LRCheck(CostComputor* cost_computor, float32 lr_check_thres, sint16* init_disp_mat, const size_t& idp_psize,
	const StereoROI_T* ste_roi)
{
	if (width_ <= 0 || height_ <= 0 || min_disparity_ >= max_disparity_ || !cost_computor) {
		return;
	}
	const sint32& disp_range = max_disparity_ - min_disparity_;

	sint32 roi_x = 0, roi_y = 0, roi_w = width_, roi_h = height_;
	if (ste_roi) {
		roi_x = ste_roi->x; roi_y = ste_roi->y;
		roi_w = ste_roi->w; roi_h = ste_roi->h;
	}
	else {
		roi_x = cu_max(0, min_disparity_); roi_y = 0;
		roi_w = width_ - roi_x; roi_h = height_;
	}

	cost_computor->ComputeCost(init_disp_mat, idp_psize, ste_roi, false);
	auto& cost_init = *(cost_computor->get_cost_ptr());

	auto streams = static_cast<cudaStream_t*>(cu_streams_);
	dim3 threads_CA(disp_range / 2, cu_max(1, THREADS_COMMON * 2 / disp_range));
	dim3 block_CA(ceil((roi_w * 1.0) / threads_CA.y), 1);
	dim3 block_CA2(ceil((roi_h * 1.0) / threads_CA.y), 1);
	if (init_disp_mat && idp_psize > 0) {
		if (disp_range == 64) {
			if (constant_p2_) {
				cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Left2Right << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Right2Left << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
			}
			else {
				cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize, img_right_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Left2Right << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize, img_right_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize, img_right_, im_psize_);
				cusgm_ca::Kernel_Aggregate_Right2Left << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize, img_right_, im_psize_);
				
			}
			for (sint32 i = 0; i < 4; i++)
				cudaStreamSynchronize(streams[i]);
			//将各个方向代价相加到最终的代价数组中,然后统计最小代价
			dim3 threads_PA(disp_range / 2, THREADS_COMMON * 2 / disp_range);
			dim3 block_PA(ceil((width_ * 1.0 - cu_max(0, min_disparity_)) / threads_PA.y), roi_h);
			cusgm_ca::Kernel_ComputeDisparityByWTA << <block_PA, threads_PA, 4 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3], 
				disp_map_r_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_, init_disp_mat, idp_psize);
		}
		else if (disp_range == 32) {
			threads_CA.x = disp_range;
			threads_CA.y = THREADS_COMMON / disp_range;
			block_CA.x = ceil((roi_w * 1.0) / threads_CA.y);
			block_CA2.x = ceil((roi_h * 1.0) / threads_CA.y);
			if (constant_p2_) {
				cusgm_ca::Kernel_Aggregate_Vertical_Warp<0> << <block_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Horizontal_Warp<0> << <block_CA2, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Vertical_Warp<1> << <block_CA, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
				cusgm_ca::Kernel_Aggregate_Horizontal_Warp<1> << <block_CA2, threads_CA, 3 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3],
					width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_, init_disp_mat, idp_psize);
			}
			for (sint32 i = 0; i < 4; i++)
				cudaStreamSynchronize(streams[i]);
			//将各个方向代价相加到最终的代价数组中,然后统计最小代价
			const sint32 n_lines = 8;
			dim3 threads_PA(disp_range, THREADS_COMMON / disp_range);
			dim3 block_PA(ceil((roi_w * 1.0) / threads_PA.y), roi_h / n_lines);
			cusgm_ca::Kernel_ComputeDisparityByWTA_Warp<n_lines> << <block_PA, threads_PA, 3 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3],
				disp_map_r_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_, init_disp_mat, idp_psize, img_left_, im_psize_);
		}
	}
	else {
		if (disp_range != 128) {
			cusgm_ca::Kernel_Aggregate_Up2Down << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Left2Right << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Down2Up << <block_CA, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Right2Left << <block_CA2, threads_CA, 4 * threads_CA.x * threads_CA.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3], width_, height_,
				min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_);
			
		}
		else {
			dim3 threadsWarp(32, cu_max(1, THREADS_COMMON / 32));
			dim3 blockWarp(ceil((roi_w * 1.0) / threadsWarp.y), 1);
			dim3 blockWarp2(ceil((roi_h * 1.0) / threadsWarp.y), 1);
			cusgm_ca::Kernel_Aggregate_Vertical_Warpfor128<0> << <blockWarp, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[0] >> > (cost_init, cost_aggr_dir_[0],
				width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Horizontal_Warpfor128<0> << <blockWarp2, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[1] >> > (cost_init, cost_aggr_dir_[1],
				width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Vertical_Warpfor128<1> << <blockWarp, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[2] >> > (cost_init, cost_aggr_dir_[2],
				width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_y + roi_h, ca_p1_, ca_p2_);
			cusgm_ca::Kernel_Aggregate_Horizontal_Warpfor128<1> << <blockWarp2, threadsWarp, 6 * threadsWarp.x * threadsWarp.y * sizeof(uint32), streams[3] >> > (cost_init, cost_aggr_dir_[3],
				width_, height_, min_disparity_, disp_range, roi_x, roi_y, roi_x + roi_w, ca_p1_, ca_p2_);
			
		}
		for (sint32 i = 0; i < 4; i++)
			cudaStreamSynchronize(streams[i]);

		//将各个方向代价相加到最终的代价数组中,然后统计最小代价
		if (disp_range != 128) {
			dim3 threads_PA(disp_range / 2, cu_max(1, THREADS_COMMON * 2 / disp_range));
			dim3 block_PA(ceil((roi_w * 1.0) / threads_PA.y), roi_h);
			cusgm_ca::Kernel_ComputeDisparityByWTA << <block_PA, threads_PA, 4 * threads_PA.x * threads_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3], 
				disp_map_r_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_);
		}
		else {
			const sint32 n_lines = 2;
			dim3 threadsWarp_PA(32, cu_max(1, THREADS_COMMON / 32));
			dim3 blockWarp_PA(ceil((roi_w * 1.0) / threadsWarp_PA.y), roi_h / n_lines);
			cusgm_ca::Kernel_ComputeDisparityByWTA_Warpfor128<n_lines> << <blockWarp_PA, threadsWarp_PA, 6 * threadsWarp_PA.x * threadsWarp_PA.y * sizeof(uint32) >> > (cost_aggr_dir_[0], cost_aggr_dir_[1], cost_aggr_dir_[2], cost_aggr_dir_[3], 
				disp_map_r_, dp_psize_, width_, height_, min_disparity_, disp_range, roi_x, roi_y, uniquess_);
		}
	}
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif

	sint32 threadsize = THREADS_COMMON;
	dim3 block_LR(ceil(width_ * 1.0 / threadsize), height_);
	cusgm_ca::Kernel_LRCheck << <block_LR, threadsize, 0 >> > (disp_map_, disp_map_r_, width_, height_, dp_psize_, lr_check_thres);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void CostAggregator::Release()
{
	cudaFree(disp_map_);
	cudaFree(disp_map_r_);
	safeFree3D(&cost_aggr_);
	for (int k = 0; k < PATH_NUM; k++) {
		safeFree3D(&cost_aggr_dir_[k]);
	}
}

cudaPitchedPtr* CostAggregator::get_cost_ptr()
{
	return &cost_aggr_;
}

float32* CostAggregator::get_disp_ptr() const
{
	return disp_map_;
}

float32* CostAggregator::get_disp_r_ptr() const
{
	return disp_map_r_;
}

size_t CostAggregator::get_disp_psize() const
{
	return dp_psize_;
}

