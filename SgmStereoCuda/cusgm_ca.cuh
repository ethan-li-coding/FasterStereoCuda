/* -*-c++-*- AD-Census - Copyright (C) 2020.
* Author	: Yingsong Li(Ethan Li) <ethan.li.whu@gmail.com>
* Github	: https://github.com/ethan-li-coding
* Describe	: header of class CrossAggregator
*/

#ifndef SGM_CUDA_COST_AGGREGATE_H_
#define SGM_CUDA_COST_AGGREGATE_H_

#include "sgm_types.h"
#include "cusgm_cc.cuh"

class CostAggregator {
public:
	CostAggregator();
	~CostAggregator();

	/**
	 * \brief 初始化
	 * \param width			影像宽
	 * \param height		影像高
	 * \param min_disparity 最小视差
	 * \param max_disparity 最大视差
	 * \return true: succeed
	 */
	bool Initialize(const sint32& width, const sint32& height, const sint32& min_disparity, const sint32& max_disparity);

	/**\brief Release */
	void Release();

	/**
	 * \brief 为代价聚合设置数据
	 * \param img_left		左视图数据
	 * \param img_right 	右视图数据
	 * \param im_psize 		对齐后的影像数据字节宽
	 * \param cost_init		初始代价
	 */
	void SetData(uint8* img_left, uint8* img_right, const size_t& im_psize, cudaPitchedPtr* cost_init);

	/**
	 * \brief 为代价聚合设置参数
	 * \param p1			// p1
	 * \param p2			// p2
	 * \param constant_p2	// 是否采用固定p2
	 * \param uniquess		// 唯一性约束阈值
	 */
	void SetParam(const float32& p1, const float32& p2, const bool& constant_p2, const float32& uniquess);

	/**\brief aggregate */
	void Aggregate(sint16* init_disp_mat = nullptr, const size_t& idp_psize = 0, const StereoROI_T* ste_roi = nullptr);

	/**\brief lr-check */
	void LRCheck(CostComputor* cost_computor, float32 lr_check_thres, sint16* init_disp_mat = nullptr, const size_t& idp_psize = 0, const StereoROI_T* ste_roi = nullptr);


	/** \brief 获取设备端聚合代价指针 */
	cudaPitchedPtr* get_cost_ptr();

	/** \brief 获取设备端视差数据指针 (左视图)*/
	float32* get_disp_ptr() const;

	/** \brief 获取设备端视差数据指针 (右视图)*/
	float32* get_disp_r_ptr() const;

	/** \brief 获取对齐后的视差图字节宽*/
	size_t get_disp_psize() const;
private:
	/** \brief 影像尺寸 */
	sint32	width_;
	sint32	height_;

	/** \brief 影像数据 */
	uint8* img_left_;
	uint8* img_right_;
	size_t im_psize_;

	/** \brief 设备端视差图 */
	float32* disp_map_;
	float32* disp_map_r_;
	size_t dp_psize_;

	/** \brief 初始代价 */
	cudaPitchedPtr* cost_init_;
	/** \brief 聚合代价 */
	cudaPitchedPtr cost_aggr_;
	/** \brief 不同方向上的聚合代价 */
	cudaPitchedPtr cost_aggr_dir_[8];

	/** \brief min_disparity */
	sint32 min_disparity_;
	/** \brief max disparity */
	sint32 max_disparity_;
	/** \brief p1 */
	float32 ca_p1_;
	/** \brief initial p2 */
	float32 ca_p2_;
	/** \brief whether p2 is constant or not */
	bool	constant_p2_;
	/** \brief uniqueness constraint threshold */
	float32 uniquess_;

	/**\brief 设备端异步流 */
	void* cu_streams_;

	/** \brief 初始化成功标志 */
	bool is_initialized_;
};

#endif
