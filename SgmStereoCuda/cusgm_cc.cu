/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li <ethan.li.whu@gmail.com>
* Describe	: implement of cusgm_cc
*/

#include "cusgm_cc.cuh"
#include <math.h>
#include "cusgm_types.h"

#define THREADS_COMMON	128

namespace cusgm_cc {

__device__ uint8 CostAdjust(sint32 tid)
{
#if 1
	return (uint8)(expf(4 * (abs(tid - LEVEL_RANGE / 2)) / (LEVEL_RANGE / 2)));
#else
	return (uint8)abs(tid - LEVEL_RANGE / 2);
#endif
}

__device__ inline uint8* GetCostPtr(cudaPitchedPtr cost, sint32 x, sint32 y, sint32 disp_range, sint32 pixels_in_pitch)
{
	const sint32 x_out_pitch = x / pixels_in_pitch;
	sint32 x_in_pitch = x - __mul24(pixels_in_pitch, x_out_pitch);
	return (uint8*)(cost.ptr) + __mul24(__mul24(y, cost.ysize), cost.pitch) + __mul24(x_out_pitch, cost.pitch) + __mul24(x_in_pitch, disp_range);
}

__global__ void Kernel_CensusCompute(uint8* __restrict__ bytes_ptr, uint32* __restrict__ census_ptr, sint32 width, sint32 height, size_t im_psize, size_t cs_psize, uint32 invalid_val)
{
	//每个线程对应一个像素，计算Census变换之后的值，存储到census_ptr矩阵中
	sint32 tdid = threadIdx.x;
	sint32 bkid_x = blockIdx.x;
	sint32 bkid_y = blockIdx.y;

	const sint32 im_psize_width = im_psize / sizeof(uint8);

	sint32 bytes_pos = bkid_y * im_psize_width + bkid_x * blockDim.x + tdid;
	sint32 image_x = bkid_x * blockDim.x + tdid;
	sint32 image_y = bkid_y;
	if (image_x >= width)
		return;

	uint32* census = (uint32*)((uint8*)census_ptr + bkid_y * cs_psize) + image_x;

	uint32 census_value = 0;
	const uint8 gray_center = bytes_ptr[bytes_pos];

	sint32 csw = 2;
	sint32 csh = 2;
	if (image_x < csw || image_x >= width - csw || image_y < csh || image_y >= height - csh){
		*census = invalid_val;
	}
	else{
		for (sint32 i = -csh; i <= csh; i++){
			for (sint32 j = -csw; j <= csw; j++){
				census_value <<= 1;
				uint8 gray = bytes_ptr[(image_y + i) * im_psize_width + image_x + j];
				if (gray < gray_center){
					census_value += 1;
				}
			}
		}
		*census = census_value;
	}
}
__global__ void Kernel_CensusCompute(uint8* __restrict__ bytes_ptr, uint64* __restrict__ census_ptr, sint32 width, sint32 height, size_t im_psize, size_t cs_psize, uint32 invalid_val)
{
	//每个线程对应一个像素，计算Census变换之后的值，存储到census_ptr矩阵中
	sint32 tdid = threadIdx.x;
	sint32 bkid_x = blockIdx.x;
	sint32 bkid_y = blockIdx.y;

	const sint32 im_psize_width = im_psize / sizeof(uint8);

	sint32 bytes_pos = bkid_y * im_psize_width + bkid_x * blockDim.x + tdid;
	sint32 image_x = bkid_x * blockDim.x + tdid;
	sint32 image_y = bkid_y;
	if (image_x >= width)
		return;

	uint64* census = (uint64*)((uint8*)census_ptr + bkid_y * cs_psize) + image_x;

	uint64 census_value = 0;
	uint8 gray_center = bytes_ptr[bytes_pos];
	sint32 csw = 4;
	sint32 csh = 3;
	if (image_x < csw || image_x >= width - csw || image_y < csh || image_y >= height - csh){
		*census = invalid_val;
	}
	else {
		sint32 count = 0;
		for (sint32 i = -csh; i <= csh; i++){
			for (sint32 j = -csw; j <= csw; j++){
				census_value <<= 1;
				uint8 gray = bytes_ptr[(image_y + i) * im_psize_width + image_x + j];
				if (gray < gray_center){
					census_value += 1;
					count++;
				}
			}
		}
		*census = census_value;
	}
}

__device__ uint8 Hamming(uint32 a, uint32 b)
{
	return __popc(a ^ b);
}
__device__ uint8 Hamming(uint64 a, uint64 b)
{
	return __popcll(a ^ b);
}


__global__ void Kernel_CostCompute(uint64* census_left, uint64* census_right, cudaPitchedPtr cost_ptr, sint32 width, sint32 height, size_t cs_psize, sint32 min_disparity, sint32 disp_range, sint32 xoffset, sint32 yoffset, bool left2right)
{
	//分配disp_range个线程，计算相邻的disp_range个像素，这样可以保证内存的载入次数更少
	if (left2right){
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y + yoffset;

		uint64* censusl = (uint64*)((uint8*)census_left + image_y * cs_psize);
		uint64* censusr = (uint64*)((uint8*)census_right + image_y * cs_psize);

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint64 sr_census_ll[];
		if (image_x + threadIdx.x < width)
			sr_census_ll[threadIdx.x] = censusl[image_x + threadIdx.x];
		sint32 image_pos_r = (image_x - min_disparity - disp_range + threadIdx.x);
		sr_census_ll[threadIdx.x + disp_range] = (image_pos_r >= 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0];
		sr_census_ll[threadIdx.x + 2 * disp_range] = (image_pos_r + disp_range >= 0 && image_pos_r + disp_range < width) ? censusr[image_pos_r + disp_range] : censusr[0];

		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
		for (sint32 i = 0; i < iEnd; i++){
			cost[threadIdx.x] = Hamming(sr_census_ll[i], sr_census_ll[2 * disp_range + i - threadIdx.x]);
			cost += costptrstep;
		}
	}
	else{
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y + yoffset;

		uint64* censusl = (uint64*)((uint8*)census_left + image_y * cs_psize);
		uint64* censusr = (uint64*)((uint8*)census_right + image_y * cs_psize);

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint64 sr_census_ll[];
		if (image_x + threadIdx.x < width)
			sr_census_ll[threadIdx.x] = censusl[image_x + threadIdx.x];
		sint32 image_pos_r = (image_x + min_disparity + threadIdx.x);
		sr_census_ll[threadIdx.x + disp_range] = (image_pos_r >= 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0];
		sr_census_ll[threadIdx.x + 2 * disp_range] = (image_pos_r + disp_range >= 0 && image_pos_r + disp_range < width) ? censusr[image_pos_r + disp_range] : censusr[0];
		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
		for (sint32 i = 0; i < iEnd; i++){
			cost[threadIdx.x] = Hamming(sr_census_ll[i], sr_census_ll[disp_range + i + threadIdx.x]);
			cost += costptrstep;
		}
	}
}
__global__ void Kernel_CostCompute(uint64* census_left, uint64* census_right, cudaPitchedPtr cost_ptr, sint32 width, sint32 height, size_t cs_psize, sint32 disp_range, sint32 xoffset, sint32 yoffset, bool left2right, sint16* init_disp_left, size_t iInitVPitchSize)
{
	//分配disp_range个线程，计算相邻的disp_range个像素，这样可以保证内存的载入次数更少
	if (left2right){
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y * blockDim.y + threadIdx.y + yoffset;
		if (image_y >= height)
			return;
		uint64* censusl = (uint64*)((uint8*)census_left + image_y * cs_psize);
		uint64* censusr = (uint64*)((uint8*)census_right + image_y * cs_psize);
		sint16* initv = (sint16*)((uint8*)init_disp_left + image_y * iInitVPitchSize);

		sint32 data_offset = threadIdx.y * blockDim.x * 3;
		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint64 sr_census_ll[];
		if (image_x + threadIdx.x < width){
			sr_census_ll[data_offset + threadIdx.x] = censusl[image_x + threadIdx.x];
			sr_census_ll[data_offset + threadIdx.x + disp_range] = initv[image_x + threadIdx.x];
		}
		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
#pragma unroll 32
		for (sint32 i = 0; i < iEnd; i++){
			sint32 image_pos_r = image_x + i - threadIdx.x - sr_census_ll[data_offset + i + disp_range];
			if (image_pos_r > 0 && image_pos_r < width)
				cost[threadIdx.x] = Hamming(sr_census_ll[data_offset + i], censusr[image_pos_r]) + CostAdjust(threadIdx.x);
			else
				cost[threadIdx.x] = 0xff;
			cost += costptrstep;
		}
	}
	else{
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y * blockDim.y + threadIdx.y + yoffset;
		if (image_y >= height)
			return;
		uint64* censusl = (uint64*)((uint8*)census_left + image_y * cs_psize);
		uint64* censusr = (uint64*)((uint8*)census_right + image_y * cs_psize);
		sint16* initv = (sint16*)((uint8*)init_disp_left + image_y * iInitVPitchSize);
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint64 sr_census_ll[];
		if (image_x + threadIdx.x < width){
			sr_census_ll[data_offset + threadIdx.x] = censusl[image_x + threadIdx.x];
			sr_census_ll[data_offset + threadIdx.x + disp_range] = initv[image_x + threadIdx.x];
		}
		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
#pragma unroll 32
		for (sint32 i = 0; i < iEnd; i++){
			sint32 image_pos_r = image_x + i + threadIdx.x + sr_census_ll[data_offset + i + disp_range];
			cost[threadIdx.x] = Hamming(sr_census_ll[data_offset + i], (image_pos_r > 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0]) + CostAdjust(threadIdx.x);
			cost += costptrstep;
		}
	}
}
__global__ void Kernel_CostCompute(uint32* census_left, uint32* census_right, cudaPitchedPtr cost_ptr, sint32 width, sint32 height, size_t cs_psize, sint32 min_disparity, sint32 disp_range, sint32 xoffset, sint32 yoffset, bool left2right)
{
	//分配disp_range个线程，计算相邻的disp_range个像素，这样可以保证内存的载入次数更少
	if (left2right){
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y + yoffset;

		uint32* censusl = (uint32*)((uint8*)census_left + image_y * cs_psize);
		uint32* censusr = (uint32*)((uint8*)census_right + image_y * cs_psize);

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint32 sr_census[];
		if (image_x + threadIdx.x < width)
			sr_census[threadIdx.x] = censusl[image_x + threadIdx.x];
		sint32 image_pos_r = (image_x - min_disparity - disp_range + threadIdx.x);
		sr_census[threadIdx.x + disp_range] = (image_pos_r > 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0];
		sr_census[threadIdx.x + 2 * disp_range] = (image_pos_r + disp_range >= 0 && image_pos_r + disp_range < width) ? censusr[image_pos_r + disp_range] : censusr[0];

		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
		for (sint32 i = 0; i < iEnd; i++){
			cost[threadIdx.x] = Hamming(sr_census[i], sr_census[2 * disp_range + i - threadIdx.x]);
			cost += costptrstep;
		}
	}
	else{
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y + yoffset;

		uint32* censusl = (uint32*)((uint8*)census_left + image_y * cs_psize);
		uint32* censusr = (uint32*)((uint8*)census_right + image_y * cs_psize);

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint32 sr_census[];
		if (image_x + threadIdx.x < width)
			sr_census[threadIdx.x] = censusl[image_x + threadIdx.x];
		sint32 image_pos_r = (image_x + min_disparity + threadIdx.x);
		sr_census[threadIdx.x + disp_range] = (image_pos_r > 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0];
		sr_census[threadIdx.x + 2 * disp_range] = (image_pos_r + disp_range >= 0 && image_pos_r + disp_range < width) ? censusr[image_pos_r + disp_range] : censusr[0];

		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
		for (sint32 i = 0; i < iEnd - 1; i++){
			cost[threadIdx.x] = Hamming(sr_census[i], sr_census[disp_range + i + threadIdx.x]);
			cost += costptrstep;
		}
	}
}
__global__ void Kernel_CostCompute(uint32* census_left, uint32* census_right, cudaPitchedPtr cost_ptr, sint32 width, sint32 height, size_t cs_psize, sint32 disp_range, sint32 xoffset, sint32 yoffset, bool left2right, sint16* init_disp_left, size_t iInitVPitchSize)
{
	//分配disp_range个线程，计算相邻的disp_range个像素，这样可以保证内存的载入次数更少
	if (left2right)	{
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y * blockDim.y + threadIdx.y + yoffset;
		if (image_y >= height)
			return;

		uint32* censusl = (uint32*)((uint8*)census_left + image_y * cs_psize);
		uint32* censusr = (uint32*)((uint8*)census_right + image_y * cs_psize);
		sint16* initv = (sint16*)((uint8*)init_disp_left + image_y * iInitVPitchSize);
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint32 sr_census[];
		if (image_x + threadIdx.x < width){
			sr_census[data_offset + threadIdx.x] = censusl[image_x + threadIdx.x];
			sr_census[data_offset + threadIdx.x + disp_range] = initv[image_x + threadIdx.x];
		}
		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
#pragma unroll 32
		for (sint32 i = 0; i < iEnd; i++){
			sint32 image_pos_r = image_x + i - threadIdx.x - sr_census[data_offset + i + disp_range];
			if (image_pos_r > 0 && image_pos_r < width)
				cost[threadIdx.x] = Hamming(sr_census[data_offset + i], censusr[image_pos_r]) + CostAdjust(threadIdx.x);
			else
				cost[threadIdx.x] = INVALID_COST;
			cost += costptrstep;
		}
	}
	else{
		sint32 image_x = blockIdx.x * blockDim.x + xoffset;
		sint32 image_y = blockIdx.y * blockDim.y + threadIdx.y + yoffset;
		if (image_y >= height)
			return;

		uint32* censusl = (uint32*)((uint8*)census_left + image_y * cs_psize);
		uint32* censusr = (uint32*)((uint8*)census_right + image_y * cs_psize);
		sint16* initv = (sint16*)((uint8*)init_disp_left + image_y * iInitVPitchSize);
		sint32 data_offset = threadIdx.y * blockDim.x * 3;

		//每个线程网格包括iDispRangle个线程，内循环处理iDispRangle个像素，这样可以充分减少全局内存的读取次数
		extern __shared__ uint32 sr_census[];
		if (image_x + threadIdx.x < width){
			sr_census[data_offset + threadIdx.x] = censusl[image_x + threadIdx.x];
			sr_census[data_offset + threadIdx.x + disp_range] = initv[image_x + threadIdx.x];
		}
		__syncthreads();

		sint32 pixels_in_pitch = cost_ptr.pitch / disp_range;
		uint8* cost = GetCostPtr(cost_ptr, image_x, image_y, disp_range, pixels_in_pitch);
		uint32 costptrstep = disp_range;
		sint32 iEnd = cu_min(disp_range, width - image_x);
#pragma unroll 32
		for (sint32 i = 0; i < iEnd; i++){
			sint32 image_pos_r = image_x + i + threadIdx.x + sr_census[data_offset + i + disp_range];
			cost[threadIdx.x] = Hamming(sr_census[data_offset + i], (image_pos_r > 0 && image_pos_r < width) ? censusr[image_pos_r] : censusr[0]) + CostAdjust(threadIdx.x);
			cost += costptrstep;
		}
	}
}

}

CostComputor::CostComputor(): width_(0), height_(0), img_left_(nullptr), img_right_(nullptr), im_psize_(0),
                                census_left_(nullptr), census_right_(nullptr), cs_psize_(0), cs_mode_(0),
                                cost_(nullptr),
                                min_disparity_(0), max_disparity_(0),                                                     
                                is_initialized_(false) { }

CostComputor::~CostComputor()
{
	Release();
}

bool CostComputor::Initialize(const sint32& width, const sint32& height, const sint32& min_disparity,
	const sint32& max_disparity, const sint32& cs_mode)
{
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	max_disparity_ = max_disparity;
	cs_mode_ = cs_mode;

	if (width_ <= 0 || height_ <= 0 || min_disparity_ >= max_disparity_) {
		is_initialized_ = false;
		return  false;
	}
	const sint32& disp_range = max_disparity_ - min_disparity_;

	// initial census mat
	if (cs_mode_ == 0) {
		// mode = 5x5
		if (!CudaSafeCall(cudaMallocPitch(&census_left_, &cs_psize_, size_t(width_) * sizeof(uint32), size_t(height_))) || 
			!CudaSafeCall(cudaMallocPitch(&census_right_, &cs_psize_, size_t(width_) * sizeof(uint32), size_t(height_)))) {
			is_initialized_ = false;
			return false;
		}
	}
	else {
		// mode = 9x7
		if (!CudaSafeCall(cudaMallocPitch(&census_left_, &cs_psize_, size_t(width_) * sizeof(uint64), size_t(height_))) || 
			!CudaSafeCall(cudaMallocPitch(&census_right_, &cs_psize_, size_t(width_) * sizeof(uint64), size_t(height_)))) {
			is_initialized_ = false;
			return false;
		}
	}

	// initial cost mat
	cost_ = new cudaPitchedPtr;
	cudaExtent extent = make_cudaExtent(disp_range, 32, 32);
	cudaPitchedPtr temp{};
	if (!CudaSafeCall(cudaMalloc3D(&temp, extent))) {
		is_initialized_ = false;
		return false;
	}
	// malloc aligned 3d array
	const auto pixel_in_pitch = cu_max(1, temp.pitch / disp_range);
	extent = make_cudaExtent(temp.pitch, (width_ + pixel_in_pitch - 1) / pixel_in_pitch, height_);
	if (!CudaSafeCall(cudaMalloc3D(cost_, extent))) {
		is_initialized_ = false;
		return false;
	}
	SafeCudaFree(temp.ptr);

	is_initialized_ = true;
	return is_initialized_;
}

void CostComputor::SetData(uint8* img_left, uint8* img_right, const size_t& im_psize)
{
	img_left_ = img_left;
	img_right_ = img_right;
	im_psize_ = im_psize;
}

cudaPitchedPtr* CostComputor::get_cost_ptr() const
{
	return cost_;
}

void CostComputor::CensusTransform(const bool& left) const
{
	if (!is_initialized_) return;

	auto thread_size = THREADS_COMMON;
	dim3 block(cs_psize_ / sizeof(uint8) / thread_size, height_);
	if (left) {
		if (cs_mode_ == 0) {
			// mode = 5x5
			cusgm_cc::Kernel_CensusCompute << <block, thread_size >> > (img_left_, static_cast<uint32*>(census_left_), width_, height_, im_psize_, cs_psize_, 0);
		}
		else {
			// mode = 9x7
			cusgm_cc::Kernel_CensusCompute << <block, thread_size >> > (img_left_, static_cast<uint64*>(census_left_), width_, height_, im_psize_, cs_psize_, 0);
		}
	}
	else {
		if (cs_mode_ == 0) {
			// mode = 5x5
			cusgm_cc::Kernel_CensusCompute << <block, thread_size >> > (img_right_, static_cast<uint32*>(census_right_), width_, height_, im_psize_, cs_psize_, 0xFFFFFFFF);
		}
		else {
			// mode = 9x7
			cusgm_cc::Kernel_CensusCompute << <block, thread_size >> > (img_right_, static_cast<uint64*>(census_right_), width_, height_, im_psize_, cs_psize_, 0xFFFFFFFF);
		}
	}
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void CostComputor::ComputeCost(sint16* init_disp_mat, const size_t& idp_psize, const StereoROI_T* ste_roi, const bool& left2right) const
{
	if (!is_initialized_) return;

	const sint32& disp_range = max_disparity_ - min_disparity_;

	sint32 roi_x = 0, roi_y = 0, roi_w = width_, roi_h = height_;
	if (ste_roi)
	{
		roi_x = ste_roi->x; roi_y = ste_roi->y;
		roi_w = ste_roi->w; roi_h = ste_roi->h;
	}
	else
	{
		roi_x = cu_max(0, min_disparity_); roi_y = 0;
		roi_w = width_ - roi_x; roi_h = height_;
	}

	dim3 thread(disp_range, cu_max(1, 64 / disp_range));
	dim3 block(ceil((roi_w * 1.0) / thread.x), ceil((roi_h * 1.0) / thread.y));

	if (cs_mode_ == 0) {
		auto* census_l = left2right? static_cast<uint32*>(census_left_): static_cast<uint32*>(census_right_);
		auto* census_r = left2right ? static_cast<uint32*>(census_right_) : static_cast<uint32*>(census_left_);
		// mode = 5x5
		if (init_disp_mat && idp_psize > 0) {
			cusgm_cc::Kernel_CostCompute << <block, thread, thread.x * 3 * thread.y * sizeof(uint32) >> > (census_l, census_r, *cost_, width_, height_, cs_psize_,
				disp_range, roi_x, roi_y, left2right, init_disp_mat, idp_psize);
		}
		else {
			cusgm_cc::Kernel_CostCompute << <block, thread, thread.x * 3 * thread.y * sizeof(uint32) >> > (census_l, census_r, *cost_, width_, height_, cs_psize_,
				min_disparity_, disp_range, roi_x, roi_y, left2right);
		}
	}
	else {
		auto* census_l = left2right ? static_cast<uint64*>(census_left_) : static_cast<uint64*>(census_right_);
		auto* census_r = left2right ? static_cast<uint64*>(census_right_) : static_cast<uint64*>(census_left_);
		// mode = 9x7
		if (init_disp_mat && idp_psize > 0) {
			cusgm_cc::Kernel_CostCompute << <block, thread, thread.x * 3 * thread.y * sizeof(uint64) >> > (census_l, census_r, *cost_, width_, height_, cs_psize_,
				disp_range, roi_x, roi_y, left2right, init_disp_mat, idp_psize);
		}
		else {
			cusgm_cc::Kernel_CostCompute << <block, thread, thread.x * 3 * thread.y * sizeof(uint64) >> > (census_l, census_r, *cost_, width_, height_, cs_psize_,
				min_disparity_, disp_range, roi_x, roi_y, left2right);
		}
	}

#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void CostComputor::SetMinDisparity(const sint32& min_disparity)
{
	const auto range = max_disparity_ - min_disparity_;
	min_disparity_ = min_disparity;
	max_disparity_ = min_disparity + range;
}

void CostComputor::Release()
{
	SafeCudaFree(census_left_);
	SafeCudaFree(census_right_);
	safeFree3D(static_cast<cudaPitchedPtr*>(cost_));
}
