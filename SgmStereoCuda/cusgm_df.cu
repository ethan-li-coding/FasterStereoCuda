/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li <ethan.li.whu@gmail.com>
* Describe	: implement of DisparityFilter
*/

#include "cusgm_df.h"
#include "cusgm_types.h"

#define THREADS_COMMON	128
#define FILTER_T_PER_BLOCK 32

namespace cusgm_df {
	__device__ void swap(float32& a, float32& b) { float32 tmp = a; a = b; b = tmp; }

	__global__ void Kernel_Median3x3(float32* disp_ptr, sint32 width, sint32 height, sint32 dp_psize)
	{
		//利用快速中值计算法进行中值滤波
		//利用共享内存，减少全局内存的读取次数
		sint32 image_x = (blockIdx.x == 0) ? (blockIdx.x * blockDim.x + threadIdx.x) : (blockIdx.x * blockDim.x - 2 * blockIdx.x + threadIdx.x);
		sint32 image_y = (blockIdx.y == 0) ? (blockIdx.y * blockDim.y + threadIdx.y) : (blockIdx.y * blockDim.y - 2 * blockIdx.y + threadIdx.y);
		if (image_x >= width || image_y >= height)
		{
			return;
		}

		float32* disp = (float32*)((uint8*)disp_ptr + image_y * dp_psize);
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

	__device__ float32 gaussR(float32 sigma, float32 dist)
	{
		return exp(-(dist * dist) / (2.0 * sigma * sigma));
	}
	__device__ float32 linearR(float32 sigma, float32 dist)
	{
		return cu_max(1.0f, cu_max(0.0f, 1.0f - (dist * dist) / (2.0 * sigma * sigma)));
	}
	__device__ float32 gaussD(float32 sigma, sint32 x, sint32 y)
	{
		return exp(-((x * x + y * y) / (2.0f * sigma * sigma)));
	}
	__device__ float32 gaussD(float32 sigma, sint32 x)
	{
		return exp(-((x * x) / (2.0f * sigma * sigma)));
	}

	__global__ void Kernel_GaussFilter(float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR, uint32 width, uint32 height, size_t pitch_size)
	{
		const sint32 x = blockIdx.x * blockDim.x + threadIdx.x;
		const sint32 y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return;

		const sint32 kernelRadius = (sint32)ceil(2.0 * sigmaD);

		d_output[y * pitch_size / sizeof(float32) + x] = INVALID_VALUE;

		float32 sum = 0.0f;
		float32 sumWeight = 0.0f;

		const float32 depthCenter = d_input[y * pitch_size / sizeof(float32) + x];
		if (depthCenter != INVALID_VALUE) {
			for (sint32 m = x - kernelRadius; m <= x + kernelRadius; m++) {
				for (sint32 n = y - kernelRadius; n <= y + kernelRadius; n++) {
					if (m >= 0 && n >= 0 && m < width && n < height) {
						const float32 currentDepth = d_input[n * pitch_size / sizeof(float32) + m];
						if (currentDepth != INVALID_VALUE && fabs(depthCenter - currentDepth) < sigmaR) {
							const float32 weight = gaussD(sigmaD, m - x, n - y);
							sumWeight += weight;
							sum += weight * currentDepth;
						}
					}
				}
			}
		}

		if (sumWeight > 0.0f) d_output[y * pitch_size / sizeof(float32) + x] = sum / sumWeight;
	}

	__global__ void Kernel_BilateralFilter(float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR, uint32 width, uint32 height, size_t pitch_size)
	{
		const sint32 x = blockIdx.x * blockDim.x + threadIdx.x;
		const sint32 y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height) return;

		const sint32 kernelRadius = (sint32)ceil(2.0 * sigmaD);

		d_output[y * pitch_size / sizeof(float32) + x] = INVALID_VALUE;

		float32 sum = 0.0f;
		float32 sumWeight = 0.0f;

		const float32 depthCenter = d_input[y * pitch_size / sizeof(float32) + x];
		if (depthCenter != INVALID_VALUE)
		{
			for (sint32 m = x - kernelRadius; m <= x + kernelRadius; m++)
			{
				for (sint32 n = y - kernelRadius; n <= y + kernelRadius; n++)
				{
					if (m >= 0 && n >= 0 && m < width && n < height)
					{
						const float32 currentDepth = d_input[n * pitch_size / sizeof(float32) + m];

						if (currentDepth != INVALID_VALUE && fabs(depthCenter - currentDepth) < sigmaR) {
							const float32 weight = gaussD(sigmaD, m - x, n - y) * gaussR(sigmaR, currentDepth - depthCenter);

							sumWeight += weight;
							sum += weight * currentDepth;
						}
					}
				}
			}

			if (sumWeight > 0.0f) d_output[y * pitch_size / sizeof(float32) + x] = sum / sumWeight;
		}
	}

	template<class T>
	__global__ void Kernel_Erosion(T* d_output, T* d_input, sint32 wndsize, sint32 width, sint32 height, size_t pitch_size)
	{
		uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
		uint32 j = blockIdx.y * blockDim.y + threadIdx.y;
		sint32 wth = (wndsize - 1) / 2;
		T in_v, out_v;
		if (i >= 0 && i < width && j >= 0 && j < height)
		{
			in_v = d_input[j * pitch_size / sizeof(T) + i];
			out_v = in_v;
			if (i > wth && i < width - wth && j > wth && j < height - wth)
			{
				for (sint32 w = -wth; w < wth + 1; w++)
				{
					for (sint32 h = -wth; h < wth + 1; h++)
					{
						if (d_input[(j + h) * pitch_size / sizeof(T) + i + w] < out_v)
							out_v = d_input[(j + h) * pitch_size / sizeof(T) + i + w];
					}
				}
			}
			if (out_v == INVALID_VALUE)
				d_output[j * pitch_size / sizeof(T) + i] = out_v;
			else
				d_output[j * pitch_size / sizeof(T) + i] = in_v;
		}
	}

	template<class T>
	__global__ void Kernel_Dilation(T* d_output, T* d_input, sint32 wndsize, sint32 width, sint32 height, size_t pitch_size)
	{
		uint32 i = blockIdx.x * blockDim.x + threadIdx.x;
		uint32 j = blockIdx.y * blockDim.y + threadIdx.y;
		sint32 wth = (wndsize - 1) / 2;
		if (i >= 0 && i < width && j >= 0 && j < height)
		{
			T out_v = d_input[j * pitch_size / sizeof(T) + i];
			if (out_v == INVALID_VALUE || out_v == INVALID_VALUE_SHORT)
			{
				if (i > wth && i < width - wth && j > wth && j < height - wth)
				{
					for (sint32 w = -wth; w < wth + 1; w++)
					{
						for (sint32 h = -wth; h < wth + 1; h++)
						{
							if (d_input[(j + h) * pitch_size / sizeof(T) + i + w] > out_v)
								out_v = d_input[(j + h) * pitch_size / sizeof(T) + i + w];
						}
					}
				}
			}
			d_output[j * pitch_size / sizeof(T) + i] = out_v;
		}
	}
}

DisparityFilter::DisparityFilter(): width_(0), height_(0),
									disp_map_(nullptr), dp_psize_(0), disp_map_filter_(nullptr), disp_map_out_(nullptr),
                                    do_median_filter_(false),
                                    postfilter_type_(), morphology_type_(), 
									is_initialized_(false) { }

DisparityFilter::~DisparityFilter()
{
	
}

bool DisparityFilter::Initialize(const sint32& width, const sint32& height)
{
	width_ = width;
	height_ = height;
	if (width_ <= 0 || height_ <= 0) {
		is_initialized_ = false;
		return  false;
	}

	if (!CudaSafeCall(cudaMallocPitch(reinterpret_cast<void**>(&disp_map_filter_), &dp_psize_, size_t(width_) * sizeof(float32), size_t(height_)))) {
		is_initialized_ = false;
		return false;
	}

	is_initialized_ = true;

	return is_initialized_;
}

void DisparityFilter::Release() const
{
	cudaFree(disp_map_filter_);
}

void DisparityFilter::SetData(float32* disp_map, const size_t& dp_psize)
{
	disp_map_ = disp_map;
	dp_psize_ = dp_psize;
}

void DisparityFilter::SetParam(bool do_median_filter, CuSGMOption::PF_Type postfilter_type, CuSGMOption::MP_Type morphology_type)
{
	do_median_filter_ = do_median_filter;
	postfilter_type_ = postfilter_type;
	morphology_type_ = morphology_type;
}

void DisparityFilter::Filter()
{
	if(!is_initialized_ || disp_map_ == nullptr || dp_psize_ <= 0) {
		return;
	}

	if (do_median_filter_) {
		//中值滤波
		Median3X3FilterCuda(disp_map_, width_, height_, dp_psize_);
	}
	
	disp_map_out_ = disp_map_;

	// 滤波后处理
	if (postfilter_type_ == CuSGMOption::PF_Type::PF_GAUSS) {
		//高斯滤波
		GaussFilterFloatCuda(disp_map_filter_, disp_map_out_, 0.5, 1.0, width_, height_, dp_psize_);
		disp_map_out_ = disp_map_filter_;
	}
	else if (postfilter_type_ == CuSGMOption::PF_Type::PF_BILATERAL) {
		//双边滤波
		BilateralFilterFloatCuda(disp_map_filter_, disp_map_out_, 0.5, 1.0, width_, height_, dp_psize_);
		disp_map_out_ = disp_map_filter_;
	}

	// 形态学处理
	if (morphology_type_ != CuSGMOption::MP_Type::MP_NONE) {
		if (disp_map_out_ == disp_map_) {
			if (morphology_type_ == CuSGMOption::MP_Type::MP_EROSION) {			// 腐蚀
				ErosionCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_DILATION) {	// 膨胀
				DilationCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_OPEN){		// 开运算
				ErosionCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
				DilationCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_CLOSE) {		// 闭运算
				DilationCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
				ErosionCuda(disp_map_filter_, disp_map_out_, 5, width_, height_, dp_psize_);
			}
			disp_map_out_ = disp_map_filter_;
		}
		else {
			if (morphology_type_ == CuSGMOption::MP_Type::MP_EROSION) {
				ErosionCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_DILATION) {
				DilationCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_OPEN) {
				ErosionCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
				DilationCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
			}
			else if (morphology_type_ == CuSGMOption::MP_Type::MP_CLOSE) {
				DilationCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
				ErosionCuda(disp_map_, disp_map_filter_, 5, width_, height_, dp_psize_);
			}
			disp_map_out_ = disp_map_;
		}
	}
}

float32* DisparityFilter::get_disp_map_out() const
{
	return disp_map_out_;
}

void DisparityFilter::Median3X3FilterCuda(float32* d_inout, sint32 width, sint32 height, const size_t& dp_psize)
{
	dim3 threads(32, THREADS_COMMON / 32);
	dim3 blocks(ceil((width - 2.0) / (threads.x - 2)), ceil((width - 2.0) / (threads.y - 2)));

	cusgm_df::Kernel_Median3x3 << <blocks, threads, threads.x* threads.y * sizeof(float32) >> > (d_inout, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::GaussFilterFloatCuda(float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR,
	uint32 width, uint32 height, size_t dp_psize)
{
	const dim3 gridSize((width + FILTER_T_PER_BLOCK - 1) / FILTER_T_PER_BLOCK, (height + FILTER_T_PER_BLOCK - 1) / FILTER_T_PER_BLOCK);
	const dim3 blockSize(FILTER_T_PER_BLOCK, FILTER_T_PER_BLOCK);

	cusgm_df::Kernel_GaussFilter << <gridSize, blockSize >> > (d_output, d_input, sigmaD, sigmaR, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::BilateralFilterFloatCuda(float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR,
	uint32 width, uint32 height, size_t dp_psize)
{
	const dim3 gridSize((width + FILTER_T_PER_BLOCK - 1) / FILTER_T_PER_BLOCK, (height + FILTER_T_PER_BLOCK - 1) / FILTER_T_PER_BLOCK);
	const dim3 blockSize(FILTER_T_PER_BLOCK, FILTER_T_PER_BLOCK);

	cusgm_df::Kernel_BilateralFilter << <gridSize, blockSize >> > (d_output, d_input, sigmaD, sigmaR, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::ErosionCuda(float32* d_output, float32* d_input, sint32 wndsize, sint32 width, sint32 height,
	size_t dp_psize)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	cusgm_df::Kernel_Erosion<float32> << <gridSize, blockSize >> > (d_output, d_input, wndsize, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::DilationCuda(float32* d_output, float32* d_input, sint32 wndsize, sint32 width, sint32 height,
	size_t dp_psize)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	cusgm_df::Kernel_Dilation<float32> << <gridSize, blockSize >> > (d_output, d_input, wndsize, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::ErosionCuda(sint16* d_output, sint16* d_input, sint32 wndsize, sint32 width, sint32 height,
	size_t dp_psize)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	cusgm_df::Kernel_Erosion<sint16> << <gridSize, blockSize >> > (d_output, d_input, wndsize, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}

void DisparityFilter::DilationCuda(sint16* d_output, sint16* d_input, sint32 wndsize, sint32 width, sint32 height,
	size_t dp_psize)
{
	dim3 blockSize(16, 16);
	dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
	cusgm_df::Kernel_Dilation<sint16> << <gridSize, blockSize >> > (d_output, d_input, wndsize, width, height, dp_psize);
#ifdef SYNCHRONIZE
	cudaDeviceSynchronize();
#endif
}
