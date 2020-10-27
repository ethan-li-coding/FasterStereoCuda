/* -*-c++-*- Copyright (C) 2020. All rights reserved
* Author	: Ethan Li
* https://github.com/ethan-li-coding
* Describe	: implement of FasterStereoCuda
*/

#include "FasterStereoCuda.h"

#include "StereoCuda.h"
#ifdef _DEBUG
#pragma comment(lib,"StereoCudad.lib")
#else
#pragma comment(lib,"StereoCuda.lib")
#endif
#include "Timer.h"
#include "types.h"
#include "trial_check.h"

#define USING_PAGE_LOCKED_MEMORY

class HierSgmCudaImpl
{
public:
	HierSgmCudaImpl();
	~HierSgmCudaImpl();

	typedef FasterStereoCuda::epi_ste_t epi_ste_t;
	typedef FasterStereoCuda::ste_opt1_t ste_opt1_t;
	typedef FasterStereoCuda::ste_opt2_t ste_opt2_t;
public:
	/**
	 * \brief 为类的使用提供状态初始化，包括成员变量的初始值复制，内存空间分配等，无相机参数，只能生成视差图
	 * \param option			输入，算法参数
	 * \return true: 成功 false: 失败
	 */
	bool Init(const ste_opt1_t& option);

	/**
	 * \brief 为类的使用提供状态初始化，包括成员变量的初始值复制，内存空间分配等
	 * \param option			输入，算法参数
	 * \return true: 成功 false: 失败
	 */
	bool Init2(const ste_opt2_t& option);
	
	/**\brief 释放匹配类所占用的资源，和初始化函数相对应，结束类使用时调用 */
	void Release();

	/**
	 * \brief 执行核线像对匹配
	 * \param bytes_left		输入，左影像数据
	 * \param bytes_right 		输入，右影像数据
	 * \param disparity_data 	输出，视差图，输出的视差图分辨率与原始分辨率保持一致，需预先分配内存，视差图中 *invalid_value*宏 为无效值
	 * \return
	 */
	bool Match(const unsigned char* bytes_left, const unsigned char* bytes_right, float* disparity_data);

	/**
	 * \brief 执行核线像对匹配
	 * \param bytes_left		输入，左影像数据
	 * \param bytes_right 		输入，右影像数据
	 * \param depth_data 		输出，深度图
	 * \return
	 */
	bool Match2(const unsigned char* bytes_left, const unsigned char* bytes_right, float* depth_data);

private:
	const unsigned char* bytes_left_;	// 左影像数据
	const unsigned char* bytes_right_;	// 右影像数据
	unsigned char** layer_bytes_left_;	// 金字塔层左影像数据
	unsigned char** layer_bytes_right_;	// 金字塔层右影像数据
	short** layer_initial_disps_;		// 存储金字塔各层视差初始值
	int num_layers_;					// 金字塔层数

	int width_;							// 影像宽
	int height_;						// 影像高
	int* layer_width_;					// 金字塔各层影像宽
	int* layer_height_;					// 金子塔各层影像高

	float** layer_disps_left_;			// 金字塔层视差数据-左影像
	float** layer_disps_right_;			// 金字塔层视差数据-右影像
	float* depth_data_;					// 深度数据

	StereoCuda** vision_stereo_;		// 立体匹配器

	bool is_print_timing_;				// 是否输出耗时日志

	bool in_trial;						// 试用期内标志
};


HierSgmCudaImpl::HierSgmCudaImpl(): bytes_left_(nullptr), bytes_right_(nullptr), layer_bytes_left_(nullptr),
                                    layer_bytes_right_(nullptr), layer_initial_disps_(nullptr), num_layers_(0),
                                    width_(0), height_(0),
                                    layer_width_(nullptr), layer_height_(nullptr),
                                    layer_disps_left_(nullptr), layer_disps_right_(nullptr),
                                    depth_data_(nullptr),
                                    vision_stereo_(nullptr), is_print_timing_(false), in_trial(false)
{
	// 判断是否在试用期内
	int trial = TrialCheck::Check(30);
	if(trial !=-1) {
		in_trial = true;
		printf("%d days remaining in the trial period!\n", trial);
	}
	else {
		in_trial = false;
		printf("Out of trial period!\n");
	}
}

HierSgmCudaImpl::~HierSgmCudaImpl()
{
	Release();
}

void _Zoom(const unsigned char* src,unsigned char* dst,int w_src,int h_src,int scale_x, int scale_y)
{
	if (scale_x == 1 && scale_y == 1) {
		memcpy(dst, src, w_src * h_src * sizeof(unsigned char));
		return;
	}

	const int wdst = w_src / scale_x;
	const int hdst = h_src / scale_y;
	const int gap0 = wdst;
	const int gap1 = scale_y * w_src;

	unsigned char* pline0 = dst;
	const unsigned char* pline1 = src;

	for (int i = 0; i < hdst; i++){
		for (int j = 0, j2 = 0; j < wdst; j++, j2 += scale_x){
			pline0[j] = pline1[j2];
		}
		pline0 += gap0;
		pline1 += gap1;
	}
}

int ComputeInitDispInNextLevel(float* cur, short* next, int width, int height)
{
	int count = 0;
	const int level_range = StereoCuda::get_level_range() >> 1;
	const float invad = StereoCuda::get_invad_float();
	const short invad_short = StereoCuda::get_invad_short();
	const int wid2 = width >> 1;
	for (int i = 0; i < height; i++) {
		int ii = i * width;
		int ii2 = (i >> 1) * wid2;
		short* nex = next + ii;
		for (int j = 0; j < width; j++) {
			const float disp = cur[ii2 + (j >> 1)];
			if (disp != invad) {
				nex[j] = static_cast<short>(2 * disp - level_range);
				count++;
			}
			else {
				nex[j] = invad_short;
			}
		}
	}
	return count;
}

//根据层号设置SGM算法参数
void SetSgmParameters(const bool& do_lr_check, const bool& do_rm_peaks, const bool& do_smooth, CuSGMOption& sgm_param, int layer_id,int num_layers)
{
	sgm_param.p1 = 8;
	sgm_param.p2_init = 32;
	sgm_param.p2_max = 64;
	sgm_param.cs_type = CuSGMOption::CS_5x5;
	sgm_param.num_paths = 4;
	sgm_param.unique_threshold = (layer_id == 0) ? 0.95f : 1.0f;
	sgm_param.lr_threshold = 1.0f;
	sgm_param.do_median_filter = true;
	sgm_param.post_filter_type = (layer_id == 0) ? (do_smooth ? CuSGMOption::PF_GAUSS : CuSGMOption::PF_NONE) : CuSGMOption::PF_NONE;
	sgm_param.using_constant_p2 = true;
	if (num_layers <= 1) {
		sgm_param.do_lr_check = do_lr_check;
		sgm_param.do_remove_peaks = do_rm_peaks;
		sgm_param.peaks_ratio_threshold = 0.0005f;
		sgm_param.morphology_type = CuSGMOption::MP_NONE;
	}
	else if (num_layers <= 2){
		sgm_param.do_lr_check = (layer_id == 0) ? false : do_lr_check;
		sgm_param.do_remove_peaks = (layer_id == 0) ? false : do_rm_peaks;
		sgm_param.peaks_ratio_threshold = (layer_id == 0) ? 0.0005f : 0.001f;
		sgm_param.morphology_type = CuSGMOption::MP_NONE;
	}
	else {
		sgm_param.do_lr_check = (layer_id < num_layers - 1) ? false : do_lr_check;
		sgm_param.do_remove_peaks = (layer_id < num_layers - 1) ? false : do_rm_peaks;
		sgm_param.peaks_ratio_threshold = (layer_id < num_layers - 1) ? 0.0005f : 0.001f;
		sgm_param.morphology_type = CuSGMOption::MP_NONE;
	}
}

bool HierSgmCudaImpl::Init(const ste_opt1_t& option)
{
	const int width = option.width;
	const int height = option.height;
	const int min_disp = option.min_disp;
	const int max_disp = option.max_disp;
	const int num_layers = option.num_layers;

	if (width == 0 || height == 0) {
		return false;
	}
	is_print_timing_ = false;
	Release();

	width_ = width;
	height_ = height;

	//每一层的数据分配、参数计算、匹配类初始化
	num_layers_ = max(1, num_layers);
	vision_stereo_ = new StereoCuda * [num_layers_]();
	layer_bytes_left_ = new unsigned char* [num_layers_]();
	layer_bytes_right_ = new unsigned char* [num_layers_]();
	layer_initial_disps_ = new short* [num_layers_]();
	layer_disps_left_ = new float* [num_layers_]();
	layer_disps_right_ = new float* [num_layers_]();
	layer_width_ = new int[num_layers_]();
	layer_height_ = new int[num_layers_]();

	for (int i = 0; i < num_layers_; i++) {
		vision_stereo_[i] = new StereoCuda;

		const int scale = static_cast<int>(pow(2.0, static_cast<double>(i)));
		const auto w = layer_width_[i] = width_ / scale;
		const auto h = layer_height_[i] = height_ / scale;
#ifdef USING_PAGE_LOCKED_MEMORY
		StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_bytes_left_[i]), w * h * sizeof(unsigned char));
		StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_bytes_right_[i]), w * h * sizeof(unsigned char));
		if (i > 0) {
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_disps_left_[i]), w * h * sizeof(float));
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_disps_right_[i]), w * h * sizeof(float));
		}
		if (i < num_layers_ - 1) {
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_initial_disps_[i]), w * h * sizeof(short));
		}
#else
		layer_bytes_left_[i] = new unsigned char[w * h]();
		layer_bytes_right_[i] = new unsigned char[w * h]();
		if (i > 0) {
			layer_disps_left_[i] = new float[w * h]();
			layer_disps_right_[i] = new float[w * h]();
		}
		if (i < num_layers_ - 1) {
			layer_initial_disps_[i] = new short[w * h]();
		}
#endif

		CuSGMOption sgm_param;
		SetSgmParameters(option.do_lr_check,option.do_rm_peaks,option.do_smooth, sgm_param, i, num_layers_);

		if (!sgm_param.using_constant_p2) {
			sgm_param.p1 = 10;
			sgm_param.p2_init = 150;
		}

		//估计视差范围
		double disparity_min = min_disp / scale, disparity_max = max_disp / scale;
		double disparity_range;
		if (i == num_layers_ - 1) {
			disparity_range = disparity_max - disparity_min;
			disparity_range = (static_cast<int>(disparity_range) + 63) / 64 * 64;
		}
		else {
			disparity_min = 0.0; disparity_max = StereoCuda::get_level_range();
			disparity_range = StereoCuda::get_level_range();
		}
		if (!vision_stereo_[i]->Init(layer_width_[i], layer_height_[i], static_cast<int>(disparity_min), static_cast<int>(disparity_range), sgm_param, is_print_timing_)) {
			return false;
		}
	}

	return true;
}

bool HierSgmCudaImpl::Init2(const ste_opt2_t& option)
{
	const int width			= option.width;
	const int height		= option.height;
	const int min_depth		= option.min_depth;
	const int max_depth		= option.max_depth;
	const int num_layers	= option.num_layers;
	const auto epi			= option.epi;

	if (width == 0 || height == 0) {
		return false;
	}

	is_print_timing_ = false;
	Release();

	width_ = width;
	height_ = height;
	num_layers_ = max(1, num_layers);
	
	//标定参数
	CamParam_T cam_scale;
	cam_scale.x0_left = epi.x0_left;
	cam_scale.y0_left = epi.y0_left;
	cam_scale.focus = epi.focus;
	cam_scale.x0_right = epi.x0_right;
	cam_scale.y0_right = epi.y0_right;
	cam_scale.baseline = epi.baseline;

	//为深度数据开辟内存
	depth_data_ = new float[width_ * height_]();

	//每一层的数据分配、参数计算、匹配类初始化
	vision_stereo_ = new StereoCuda * [num_layers_]();
	layer_bytes_left_ = new unsigned char* [num_layers_]();
	layer_bytes_right_ = new unsigned char* [num_layers_]();
	layer_initial_disps_ = new short* [num_layers_]();
	layer_disps_left_ = new float* [num_layers_]();
	layer_disps_right_ = new float* [num_layers_]();
	layer_width_ = new int[num_layers_]();
	layer_height_ = new int[num_layers_]();

	for (int i = 0; i < num_layers_; i++) {
		vision_stereo_[i] = new StereoCuda;

		const int scale = static_cast<int>(pow(2.0, static_cast<double>(i)));
		const int w = layer_width_[i] = width_ / scale;
		const int h = layer_height_[i] = height_ / scale;
#ifdef USING_PAGE_LOCKED_MEMORY
		StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_bytes_left_[i]), w * h * sizeof(unsigned char));
		StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_bytes_right_[i]), w * h * sizeof(unsigned char));
		if (i > 0) {
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_disps_left_[i]), w * h * sizeof(float));
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_disps_right_[i]), w *h* sizeof(float));
		}
		if (i < num_layers_ - 1) {
			StereoCuda::MallocPageLockedPtr(reinterpret_cast<void**>(&layer_initial_disps_[i]), w *h* sizeof(short));
		}
#else
		layer_bytes_left_[i] = new unsigned char[w * h]();
		layer_bytes_right_[i] = new unsigned char[w * h]();
		if (i > 0) {
			layer_disps_left_[i] = new float[w * h]();
			layer_disps_right_[i] = new float[w * h]();
		}
		if (i < num_layers_ - 1) {
			layer_initial_disps_[i] = new short[w * h]();
		}
#endif

		CuSGMOption sgm_param;
		SetSgmParameters(option.do_lr_check, option.do_rm_peaks, option.do_smooth, sgm_param, i, num_layers_);
		
		//计算视差范围
		const CamParam_T cam_param = cam_scale.Scale(scale);
		double disparity_min, disparity_max;
		double disparity_range;
		if (i == num_layers_ - 1) {
			disparity_min = cam_param.baseline * cam_param.focus / max_depth - cam_param.x0_right + cam_param.x0_left;
			disparity_max = cam_param.baseline * cam_param.focus / min_depth - cam_param.x0_right + cam_param.x0_left;
			disparity_range = disparity_max - disparity_min;
			int k = 6;
			while (pow(2, k) < disparity_range - 5)
				k++;
			disparity_range = pow(2, k);
		}
		else {
			disparity_min = 0.0; disparity_max = StereoCuda::get_level_range();
			disparity_range = StereoCuda::get_level_range();
		}

		// 初始化stereo
		if (!vision_stereo_[i]->Init2(layer_width_[i], layer_height_[i], int(disparity_min), static_cast<int>(disparity_range), sgm_param, cam_param, is_print_timing_)) {
			return false;
		}
	}

	if (is_print_timing_)
		printf("Init completed.\n\n");
	return true;
}

bool HierSgmCudaImpl::Match(const unsigned char * bytes_left, const unsigned char * bytes_right, float * disparity_data)
{
	if (!in_trial) {
		printf("Out of trial period!\n");
		return false;
	}
	if (width_ <= 0 || height_ <= 0){
		if (is_print_timing_)
			printf("image size error!\n");
		return false;
	}
	if (bytes_left == nullptr || bytes_right == nullptr || disparity_data == nullptr){
		if (is_print_timing_)
			printf("param error!\n");
		return false;
	}

	if (is_print_timing_) {
		printf("\nBegin...\n");
	}

	//变量赋值
	bytes_left_ = bytes_left;
	bytes_right_ = bytes_right;

	//使区域无效
	MyTimer timer;
	timer.Start();
	for (int i = 0; i < num_layers_; i++){
		if (i == 0){
			_Zoom(bytes_left_, layer_bytes_left_[i], width_, height_, 1, 1);
			_Zoom(bytes_right_, layer_bytes_right_[i], width_, height_, 1, 1);
		}
		else{
			const int scale = static_cast<int>(pow(2.0, static_cast<double>(i - 1)));
			const int layer_width = width_ / scale;
			const int layer_height = height_ / scale;
			_Zoom(layer_bytes_left_[i - 1], layer_bytes_left_[i], layer_width, layer_height, 2, 2);
			_Zoom(layer_bytes_right_[i - 1], layer_bytes_right_[i], layer_width, layer_height, 2, 2);
		}
	}
	timer.End();

	if (is_print_timing_) {
		printf("Zoom: %.1lf ms\n", timer.GetDurationMs());
	}

	bool result = false;

	//分层匹配，每一层的视差结果估计出下一层的像素初始视差，传入下一层
	double match_time = 0.0;
	for (int i = num_layers_ - 1; i >= 0; i--){
		timer.Start();
		if (i > 0)
			vision_stereo_[i]->Match(layer_bytes_left_[i], layer_bytes_right_[i], layer_disps_left_[i],nullptr, layer_initial_disps_[i]);
		else{
			result = vision_stereo_[i]->Match(layer_bytes_left_[i], layer_bytes_right_[i], disparity_data, nullptr, layer_initial_disps_[i]);
		}
		timer.End();
		const double time_level = timer.GetDurationMs();
		if (is_print_timing_) {
			printf("	Level %d: %.1lf ms\n", i, time_level); match_time += time_level;
		}
		//计算下一层像素的初始视差值
		if (i > 0){
			timer.Start();
			int count = ComputeInitDispInNextLevel(layer_disps_left_[i], layer_initial_disps_[i - 1], layer_width_[i - 1], layer_height_[i - 1]);
			timer.End();
			double time_level = timer.GetDurationMs();
			if (is_print_timing_) {
				printf("	EstInit %d: %.1lf ms\n", i, time_level);
			}
		}
	}
	
	if (is_print_timing_) {
		printf("Match: %.1lf ms\n", match_time);
	}

	return result;
}

bool HierSgmCudaImpl::Match2(const unsigned char* bytes_left, const unsigned char* bytes_right, float* depth_data)
{
	if (!in_trial) {
		printf("Out of trial period!\n");
		return false;
	}

	if (width_ <= 0 || height_ <= 0) {
		if (is_print_timing_)
			printf("image size error!\n");
		return false;
	}
	if (bytes_left == nullptr || bytes_right == nullptr || depth_data == nullptr) {
		if (is_print_timing_)
			printf("param error!\n");
		return false;
	}

	if (is_print_timing_)
		printf("Begin...\n");

	//变量赋值
	bytes_left_ = bytes_left;
	bytes_right_ = bytes_right;

	MyTimer timer_match, timer;
	timer.Start();

	for (int i = 0; i < num_layers_; i++) {
		if (i == 0) {
			_Zoom(bytes_left_, layer_bytes_left_[i], width_, height_, 1, 1);
			_Zoom(bytes_right_, layer_bytes_right_[i], width_, height_, 1, 1);
		}
		else {
			const int scale = static_cast<int>(pow(2.0, static_cast<double>(i - 1)));
			const int layer_width = width_ / scale;
			const int layer_height = height_ / scale;
			_Zoom(layer_bytes_left_[i - 1], layer_bytes_left_[i], layer_width, layer_height, 2, 2);
			_Zoom(layer_bytes_right_[i - 1], layer_bytes_right_[i], layer_width, layer_height, 2, 2);
		}
	}

	timer.End();
	if (is_print_timing_)
		printf("Zoom: %.1lf ms\n", timer.GetDurationMs());

	bool result = false;
	double match_time = 0.0;

	bool is_using_roi = false;
	timer_match.Start();

	//分层匹配，每一层的视差结果估计出下一层的像素初始视差，传入下一层
	StereoROI_T* stereo_roi_left = nullptr, * stereo_roi_right = nullptr;
	for (int i = num_layers_ - 1; i >= 0; i--) {
		timer.Start();
		if (i > 0)
			vision_stereo_[i]->Match(layer_bytes_left_[i], layer_bytes_right_[i], layer_disps_left_[i], is_using_roi ? layer_disps_right_[i] : nullptr, layer_initial_disps_[i], stereo_roi_left, stereo_roi_right);
		else {
			result = vision_stereo_[i]->Match2(layer_bytes_left_[i], layer_bytes_right_[i], depth_data_, layer_initial_disps_[i], stereo_roi_left, stereo_roi_right);
			if (depth_data != depth_data_) {
				memcpy(depth_data, depth_data_, width_ * height_ * sizeof(float));
			}
		}

		timer.End();
		double time_level = timer.GetDurationMs();
		if (is_print_timing_)
			printf("	Level %d: %.1lf ms\n", i, time_level);

		if (is_using_roi) {
			if (!stereo_roi_left) stereo_roi_left = new StereoROI_T();
			if (!stereo_roi_right) stereo_roi_right = new StereoROI_T();
		}

		//计算下一层像素的初始视差值以及ROI区
		if (i > 0) {
			timer.Start();
			int count = ComputeInitDispInNextLevel(layer_disps_left_[i], layer_initial_disps_[i - 1], layer_width_[i - 1], layer_height_[i - 1]);
			timer.End();
			time_level = timer.GetDurationMs();
			if (is_print_timing_)
				printf("	EstInit %d: %.1lf ms\n", i, time_level);

			if (is_using_roi) {
				timer.Start();
				StereoCuda::GetRoiFromDispMap(layer_disps_left_[i], layer_width_[i], layer_height_[i], *stereo_roi_left);
				stereo_roi_left->x *= 2; stereo_roi_left->y *= 2; stereo_roi_left->w *= 2; stereo_roi_left->h *= 2;
				StereoCuda::GetRoiFromDispMap(layer_disps_right_[i], layer_width_[i], layer_height_[i], *stereo_roi_right);
				stereo_roi_right->x *= 2; stereo_roi_right->y *= 2; stereo_roi_right->w *= 2; stereo_roi_right->h *= 2;
				timer.End();
				time_level = timer.GetDurationMs();
				if (is_print_timing_)
					printf("	GetROI %d: %.1lf ms\n", i, time_level);
			}
		}

	}
	if (is_using_roi) {
		SafeDelete(stereo_roi_left);
		SafeDelete(stereo_roi_right);
	}

	timer_match.End();
	match_time = timer_match.GetDurationMs();
	if (is_print_timing_)
		printf("Match: %.1lf ms\n", match_time);

	return result;
}


void HierSgmCudaImpl::Release()
{	
	if (num_layers_ > 0){
		for (int i = 0; i < num_layers_; i++){
			if (vision_stereo_){
				if (vision_stereo_[i]){
					vision_stereo_[i]->Release();
					SafeDelete(vision_stereo_[i]);
				}
			}
#ifdef USING_PAGE_LOCKED_MEMORY
			if (layer_bytes_left_[i]) {
				StereoCuda::FreePageLockedPtr(layer_bytes_left_[i]);
			}
			if (layer_bytes_right_[i]) {
				StereoCuda::FreePageLockedPtr(layer_bytes_right_[i]);
			}
			if (layer_initial_disps_[i]) {
				StereoCuda::FreePageLockedPtr(layer_initial_disps_[i]);
			}
			if (layer_disps_left_[i]) {
				StereoCuda::FreePageLockedPtr(layer_disps_left_[i]);
			}
			if (layer_disps_right_[i]) {
				StereoCuda::FreePageLockedPtr(layer_disps_right_[i]);
			}
#else
			if (layer_bytes_left_[i]){
				SafeDeleteArray(layer_bytes_left_[i]);
			}
			if (layer_bytes_right_[i]){
				SafeDeleteArray(layer_bytes_right_[i]);
			}
			if (layer_initial_disps_[i]){
				SafeDeleteArray(layer_initial_disps_[i]);
			}
			if (layer_disps_left_[i]){
				SafeDeleteArray(layer_disps_left_[i]);
			}
			if (layer_disps_right_[i]){
				SafeDeleteArray(layer_disps_right_[i]);
			}
#endif
		}
	}
	SafeDelete(vision_stereo_);
	SafeDelete(layer_bytes_left_);
	SafeDelete(layer_bytes_right_);
	SafeDelete(layer_initial_disps_);
	SafeDelete(layer_disps_left_);
	SafeDelete(layer_disps_right_);
	SafeDeleteArray(layer_width_);
	SafeDeleteArray(layer_height_);
	SafeDeleteArray(depth_data_);
}

FasterStereoCuda::FasterStereoCuda()
	:impl_(nullptr)
{
	impl_ = new HierSgmCudaImpl();
}

FasterStereoCuda::~FasterStereoCuda()
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	SafeDelete(impl);
}


bool FasterStereoCuda::Init(const ste_opt1_t& option) const
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	if (impl) {
		return impl->Init(option);
	}
	else {
		return false;
	}
}

bool FasterStereoCuda::Init2(const ste_opt2_t& option) const
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	if (impl) {
		return impl->Init2(option);
	}
	else {
		return false;
	}
}

bool FasterStereoCuda::Match(const unsigned char * bytes_left, const unsigned char * bytes_right, float * disparity_data) const
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	if (impl) {
		return impl->Match(bytes_left, bytes_right, disparity_data);
	}
	else {
		return false;
	}
}

bool FasterStereoCuda::Match2(const unsigned char* bytes_left, const unsigned char* bytes_right, float* depth_data) const
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	if (impl) {
		return impl->Match2(bytes_left, bytes_right, depth_data);
	}
	else {
		return false;
	}
}

void FasterStereoCuda::Release() const
{
	auto impl = static_cast<HierSgmCudaImpl*>(impl_);
	if (impl){
		impl->Release();
	}
}

bool FasterStereoCuda::MallocPageLockedPtr(void** ptr, size_t size)
{
	return StereoCuda::MallocPageLockedPtr(ptr, size);
}

bool FasterStereoCuda::FreePageLockedPtr(void* ptr)
{
	return StereoCuda::FreePageLockedPtr(ptr);
}
