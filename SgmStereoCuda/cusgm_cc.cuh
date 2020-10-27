/* -*-c++-*- SemiGlobalMatching - Copyright (C) 2020.
* Author	: Yingsong Li <ethan.li.whu@gmail.com>
* Describe	: header of cusgm_cc
*/

#ifndef SGM_CUDA_COST_COMPUTOR_H_
#define SGM_CUDA_COST_COMPUTOR_H_
#include "sgm_types.h"
#include "cuda_runtime.h"

struct cudaPitchedPtr;

class CostComputor {
public:
	CostComputor();
	~CostComputor();

	/**
	 * \brief 初始化
	 * \param width			影像宽
	 * \param height		影像高
	 * \param min_disparity	最小视差
	 * \param max_disparity	最大视差
	 * \param cs_mode		Census窗口模式 0:5x5 1:9x7
	 * \return true: succeed
	 */
	bool Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity, const sint32& cs_mode = 0);

	/** \brief release memory */
	void Release() const;

	/**
	 * \brief 为代价计算设置数据
	 * \param img_left		左视图数据
	 * \param img_right		右视图数据
	 * \param im_psize		对齐后的影像数据字节宽
	 */
	void SetData(uint8* img_left, uint8* img_right,const size_t& im_psize);

	/**
	 * \brief Census Transform
	 * \param left			true for computing left view; false for right view 
	 */
	void CensusTransform(const bool& left = true) const;

	/**
	 * \brief 代价计算
	 * \param init_disp_mat	初始视差图（从金字塔上层传递）
	 * \param idp_psize		对齐后的视差图字节宽
	 * \param ste_roi		ROI区域
	 * \param left2right	true for computing left to right; false for computing right to view,
	 */
	void ComputeCost(sint16* init_disp_mat = nullptr, const size_t& idp_psize = 0, const StereoROI_T* ste_roi = nullptr, const bool& left2right = true) const;

	/** \brief 获取设备端初始代价指针 */
	cudaPitchedPtr* get_cost_ptr() const;
private:
	/** \brief 影像尺寸 */
	sint32	width_;
	sint32	height_;

	/** \brief 影像数据 */
	uint8* img_left_;
	uint8* img_right_;
	size_t im_psize_;

	/** \brief 左视图Census变换值 */
	void* census_left_;
	/** \brief 右视图Census变换值 */
	void* census_right_;
	/** \brief Census变换值数组对齐后的字节宽	*/
	size_t cs_psize_;
	/** \brief census transform mode : 0:5x5 1:9x7 */
	sint32 cs_mode_;

	/** \brief 初始代价 */
	cudaPitchedPtr* cost_;

	/** \brief min disparity */
	sint32 min_disparity_;
	/** \brief max disparity */
	sint32 max_disparity_;

	/** \brief 初始化成功标志 */
	bool is_initialized_;
};

#endif
