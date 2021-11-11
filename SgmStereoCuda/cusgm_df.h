/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li <ethan.li.whu@gmail.com>
* Describe	: header of DisparityFilter
*/

#ifndef SGM_CUDA_DISP_FILTER_H_
#define SGM_CUDA_DISP_FILTER_H_

#include "sgm_types.h"

class DisparityFilter {
public:
	DisparityFilter();
	~DisparityFilter();

	/**
	 * \brief 初始化
	 * \param width		影像宽
	 * \param height	影像高
	 * \return
	 */
	bool Initialize(const sint32& width, const sint32& height);
	
	/** \brief 释放内存 */
	void Release();

	/**
	 * \brief 设置数据（设备端数据）
	 * \param disp_map	视差图
	 * \param im_psize	对齐后的图像pitch大小
	 * \param dp_psize	对齐后的深度图pitch大小
	 */
	void SetData(uint8* img_bytes, float32* disp_map, const size_t& im_psize, const size_t& dp_psize);

	/**
	 * \brief 设置参数
	 * \param do_median_filter	// 执行中值滤波 
	 * \param postfilter_type	// 滤波后处理类型
	 * \param morphology_type	// 形态学处理类型
	 */
	void SetParam(bool do_median_filter, CuSGMOption::PF_Type postfilter_type, CuSGMOption::MP_Type morphology_type);

	/** \brief 执行滤波 */
	void Filter();

	/** \brief 获取输出的视差图 */
	float32* get_disp_map_out() const;

public:
	// 3x3中值滤波
	static void Median3X3FilterCuda(float32* d_inout, sint32 width, sint32 height, const size_t& dp_psize);

	// 后处理滤波方法集
	static void GaussFilterFloatCuda(float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR, uint32 width, uint32 height, size_t dp_psize);
	static void BilateralFilterFloatCuda(uint8* img_bytes, float32* d_output, float32* d_input, float32 sigmaD, float32 sigmaR, uint32 width, uint32 height, size_t im_psize, size_t dp_psize);

	// 形态学处理方法集
	static void ErosionCuda(float32* d_output, float32* d_input, sint32 wndsize, sint32 width, sint32 height, size_t dp_psize);
	static void DilationCuda(float32* d_output, float32* d_input, sint32 wndsize, sint32 width, sint32 height, size_t dp_psize);
	static void ErosionCuda(sint16* d_output, sint16* d_input, sint32 wndsize, sint32 width, sint32 height, size_t dp_psize);
	static void DilationCuda(sint16* d_output, sint16* d_input, sint32 wndsize, sint32 width, sint32 height, size_t dp_psize);

private:
	/** \brief 影像尺寸*/
	sint32	width_;
	sint32	height_;

	/** \brief 左图像数据*/
	uint8*	img_bytes_;
	/** \brief 对齐的图像pitch大小*/
	size_t im_psize_;
	/** \brief 视差图*/
	float32* disp_map_;
	/** \brief 对齐的视差图pitch大小*/
	size_t dp_psize_;
	/** \brief 滤波视差图*/
	float32* disp_map_filter_;
	/** \brief 输出视差图*/
	float32* disp_map_out_;

	/** \brief 中值滤波开关*/
	bool do_median_filter_;
	/** \brief 后处理滤波开关*/
	CuSGMOption::PF_Type postfilter_type_;
	/** \brief 形态学处理开关*/
	CuSGMOption::MP_Type morphology_type_;

	/** \brief 是否初始化成功 */
	bool is_initialized_;
};

#endif
