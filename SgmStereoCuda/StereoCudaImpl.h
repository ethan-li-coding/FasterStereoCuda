#pragma once
#include "sgm_types.h"

class CostComputor;
class CostAggregator;
class DisparityFilter;
class StereoCudaImpl
{
public:
	StereoCudaImpl();
	~StereoCudaImpl();

public:
	/**
	 * \brief 初始化
	 * \param width				影像宽
	 * \param height			影像高
	 * \param min_disparity		最小视差值
	 * \param disp_range		视差范围
	 * \param sgm_option		SGM参数
	 * \param print_log			是否打印log信息
	 * \return true-成功 false-失败
	 */
	bool Init(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option, bool print_log = false);

	/**
	 * \brief 匹配
	 * \param img_left			左影像数据
	 * \param img_right			右影像数据
	 * \param disp_left			输出左视差图，尺寸与影像尺寸一致，需预先分配内存
	 * \param disp_right		输出右视差图，尺寸与影像尺寸一致，需预先分配内存
	 * \param init_disp_left	初始视差值数组，若为nullptr，则不采用初始视差值
	 * \param ste_roi_left		左影像ROI
	 * \param ste_roi_right		右影像ROI
	 * \return
	 */
	bool Match(uint8* img_left, uint8* img_right, float32* disp_left, float32* disp_right = nullptr, sint16* init_disp_left = nullptr, StereoROI_T* ste_roi_left = nullptr, StereoROI_T* ste_roi_right = nullptr);

	/**
	 * \brief 初始化2
	 * \param width				影像宽
	 * \param height 			影像高
	 * \param min_disparity 	最小视差值
	 * \param disp_range 		视差范围
	 * \param sgm_option 		SGM参数
	 * \param cam_param 		相机参数
	 * \param print_log 		是否打印log信息
	 * \return
	 */
	bool Init2(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option, CamParam_T cam_param, bool print_log = false);

	/**
	 * \brief 匹配2
	 * \param img_left			左影像数据
	 * \param img_right 		右影像数据
	 * \param depth_left 		左影像深度图，尺寸与影像尺寸一致，需预先分配内存
	 * \param init_disp_left 	左影像初始视差值数组，若为nullptr，则不采用初始视差值
	 * \param ste_roi_left		左影像ROI
	 * \param ste_roi_right		右影像ROI
	 * \return true-成功 false-失败
	 */
	bool Match2(uint8* img_left, uint8* img_right, float32* depth_left, sint16* init_disp_left = nullptr, StereoROI_T* ste_roi_left = nullptr, StereoROI_T* ste_roi_right = nullptr);


	/**
	 * \brief 获取视差图的ROI区
	 * \param disp_ptr			视差图指针
	 * \param width				视差图宽
	 * \param height			视差图高
	 * \param ste_roi			输出的ROI数据
	 */
	static void GetRoiFromDispMap(float32* disp_ptr, sint32 width, sint32 height, StereoROI_T& ste_roi);

	/**\brief 释放内存 */
	void Release();

	/**\brief 获取无效值 */
	static float32 get_invad_float();
	static sint16 get_invad_short();

	/**\brief 获取金字塔匹配模式下除最高层之外的所有层的视差搜索范围 */
	static sint16 get_level_range();

	/**\brief 在主机端分配锁页内存（传输更快） */
	static bool MallocPageLockedPtr(void** ptr, size_t size);

	/**\brief 释放主机端锁页内存 */
	static bool FreePageLockedPtr(void* ptr);

#ifdef _DEBUG
	void OutInitValueR(sint16* pInitValue);
#endif

private:
	/**\brief 代价计算 */
	bool ComputeCost(uint8* img_left, uint8* img_right, sint16* init_disp_left = nullptr) const;

	/**\brief 代价聚合 */
	bool CostAggregate(sint16* init_disp_left = nullptr, StereoROI_T* ste_roi_left = nullptr, StereoROI_T* ste_roi_right = nullptr) const;

	/**\brief 滤波 */
	void Filter() const;

	/**\brief 剔除小连通区 */
	void RemovePeaks(StereoROI_T* ste_roi_left, float32* disp_left);

private:
	/**\brief 设备端数据指针 */
	uint8* cu_img_left_;
	uint8* cu_img_right_;
	/**\brief 设备端视差图指针 */
	float32* cu_disp_out_;
	/**\brief 设备端深度图指针 */
	float32* cu_depth_left_;
	/**\brief 设备端初始视差图指针 */
	sint16* cu_inidisp_left_;
	sint16* cu_inidisp_right_;
	sint16* cu_inidisp_tmp_;

	/**\brief 主机端初始视差图指针 */
	sint16* inidisp_tmp_;

	/**\brief sgm参数 */
	CuSGMOption sgm_option_;
	/**\brief 相机参数 */
	CamParam_T cam_param_;

	/**\brief 最小视差值 */
	sint32 min_disparity_;
	/**\brief 视差范围 */
	sint32 disp_range_;

	/**\brief 影像尺寸 */
	sint32 width_;
	sint32 height_;
	size_t im_psize_;
	size_t dp_psize_;
	size_t idp_psize_;

	/**\brief 代价计算器 */
	CostComputor* computor_;
	/**\brief 代价聚合器 */
	CostAggregator* aggregator_;
	/**\brief 视差图滤波器 */
	DisparityFilter* filter_;

	/**\brief 主机端视差图指针 */
	float32* disp_ptr_;
	/**\brief 剔除小连通区并行参数结构变量 */
	void** remove_peaks_;
	/**\brief 剔除小连通区并行线程数 */
	sint32 num_threads_;

	/**\brief 设备端异步流 */
	void* cu_streams_;

	/**\brief 日志打印开关 */
	bool print_log_;
};