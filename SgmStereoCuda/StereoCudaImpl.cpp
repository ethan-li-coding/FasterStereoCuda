#include "StereoCudaImpl.h"
#include <cstdio>
#include <unordered_map>

#include "cusgm_types.h"
#include "cusgm_cc.cuh"
#include "cusgm_ca.cuh"
#include "cusgm_df.h"

#include "cusgm_util.cuh"
#include "thread.h"
#include <thread>
#include <algorithm>
#include <chrono>
using namespace std::chrono;

#define USING_PAGE_LOCKED_MEMORY

StereoCudaImpl::StereoCudaImpl() :	cu_img_left_(nullptr), cu_img_right_(nullptr), cu_disp_out_(nullptr), cu_depth_left_(nullptr),
									cu_inidisp_left_(nullptr), cu_inidisp_right_(nullptr), cu_inidisp_tmp_(nullptr), inidisp_tmp_(nullptr),
									cam_param_(), min_disparity_(0), disp_range_(0),
									width_(0), height_(0), im_psize_(0), dp_psize_(0), idp_psize_(0),
									computor_(nullptr), aggregator_(nullptr), filter_(nullptr),
									disp_ptr_(nullptr), remove_peaks_(nullptr), num_threads_(0),
									cu_streams_(nullptr), print_log_(false) {}

StereoCudaImpl::~StereoCudaImpl()
{
	Release();
}

sint32 RemovePeaks(float32* disp_map, sint32* segid, sint32 w, sint32 h, sint32 threshold)
{
	std::vector<std::vector<sint32>> segment;
	int in[4] = { -1,0,0,1 };
	int jn[4] = { 0,-1,1,0 };
	std::vector<sint32> seg;
	seg.reserve(w * h);
	sint32 segs = 0;
	sint32 s = 0;
	for (sint32 i = 0, xy = 0; i < h; i++) {
		for (sint32 j = 0; j < w; j++, xy++) {
			if (*(disp_map + xy) == INVALID_VALUE) continue;
			s++;
			const sint32 id = *(segid + xy);
			if (id == 0) {
				segs++;
				*(segid + xy) = segs;
				seg.clear();
				seg.push_back(xy);
				auto count1 = 0, count2 = 0;
				auto search_idx(0), ii(0), jj(0);
				do {
					count2 = seg.size();
					for (sint32 k = count1; k < count2; k++) {
						const sint32 base_idx = seg[k];
						ii = base_idx / w;
						jj = base_idx - ii * w;
						const float32 base = *(disp_map + base_idx);
						for (int n = 0; n < 4; n++) {
							const int ix = ii + in[n];
							const int jx = jj + jn[n];
							if (ix < 0 || ix >= h || jx < 0 || jx >= w)
								continue;
							search_idx = ix * w + jx;
							if (*(segid + search_idx) != 0)
								continue;
							const float32 search = *(disp_map + search_idx);
							if (search == INVALID_VALUE)
								continue;
							if (abs(search - base) <= 1.0f) {
								seg.push_back(search_idx);
								*(segid + search_idx) = segs;
							}
						}
					}
					count1 = count2;
				} while (count1 < seg.size());
				segment.push_back(seg);
			}
		}
	}
	const sint32 num_seg = segment.size();
	sint32 peaks = 0;
	sint32 sum = 0;
	for (sint32 k = 0; k < num_seg; k++) {
		const sint32 size = segment[k].size();
		if (size < threshold) {
			for (sint32 n = 0; n < size; n++) {
				*(disp_map + segment[k][n]) = INVALID_VALUE;
				peaks++;
			}
		}
		sum += size;
	}
	return peaks;
}

typedef class ThreadRemovePeaks: public ThreadBase {
public:
	float32* disp_ptr;
	sint32* segid_ptr;
	sint32 w, h;
	sint32 threshold;
	ThreadRemovePeaks(): ThreadBase(), disp_ptr(nullptr), segid_ptr(nullptr), w(0), h(0), threshold(0) {
		std::thread t(Run, this);
		t.detach();
	}

	~ThreadRemovePeaks() {
		WaitForTerminate();
		if (segid_ptr) { delete[] segid_ptr; segid_ptr = nullptr; }
		disp_ptr = nullptr;
	}

	static void Run(void* p) {
		auto* rmp = static_cast<ThreadRemovePeaks*>(p);
		while(rmp->running) {
			rmp->WaitToStart();
			if(rmp->disp_ptr&&rmp->segid_ptr)
				RemovePeaks(rmp->disp_ptr, rmp->segid_ptr, rmp->w, rmp->h, rmp->threshold);
			rmp->End();
		}
	}
}thread_remove_peaks;


bool StereoCudaImpl::Init(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option, bool print_log)
{
	print_log_ = print_log;

	if (print_log_)
		printf("Cuda Stereo\n");
	if (print_log_) printf("Initialize CUDA...\n");

	cudaDeviceProp device_prop{};
	sint32 num_dev = 0;
	if (!CudaSafeCall(cudaGetDeviceCount(&num_dev))) {
		return false;
	}
	if (num_dev == 0) {
		if (print_log_) printf("No CUDA-enabled device detected!\n");
		return false;
	}
	//Select the best GPU
	sint32 choose_id = 0;
	if (num_dev == 1) {
		choose_id = 0;
	}
	else {
		sint32 max_version = 0;
		for (sint32 i = 0; i < num_dev; i++) {
			if (!CudaSafeCall(cudaGetDeviceProperties(&device_prop, i))) {
				return false;
			}
			sint32 version = device_prop.major * 0x10 + device_prop.minor;
			if (max_version < version) {
				max_version = version;
				choose_id = i;
			}
		}
	}
	if (!CudaSafeCall(cudaSetDevice(choose_id))) {
		return false;
	}

	if (!CudaSafeCall(cudaGetDeviceProperties(&device_prop, choose_id))) {
		return false;
	}

	if (print_log_) printf("GPU Name: %s\n", device_prop.name);

	sgm_option_ = sgm_option;
	width_ = width;
	height_ = height;
	min_disparity_ = min_disparity;
	disp_range_ = disp_range;
	if (width_ == 0 || height_ == 0)
	{
		if (print_log_) printf("Image size error!\n");
		return false;
	}

	if (print_log_) printf("\nw = %d h = %d d = %d\n", width_, height_, disp_range_);

	if (width_ == 0 || height_ == 0 || disp_range_ == 0) {
		if (print_log_) printf("Parameters error!\n");
		return false;
	}

	if (!CudaSafeCall(cudaMallocPitch((void**)&cu_img_left_, &im_psize_, size_t(width_) * sizeof(uint8), size_t(height_)))) {
		return false;
	}
	if (!CudaSafeCall(cudaMallocPitch((void**)&cu_img_right_, &im_psize_, size_t(width_) * sizeof(uint8), size_t(height_)))) {
		return false;
	}
	if (!CudaSafeCall(cudaMallocPitch((void**)&cu_inidisp_left_, &idp_psize_, size_t(width_) * sizeof(sint16), size_t(height_)))) {
		return false;
	}
	if (sgm_option_.do_lr_check) {
		if (!CudaSafeCall(cudaMallocPitch((void**)&cu_inidisp_right_, &idp_psize_, size_t(width_) * sizeof(sint16), size_t(height_)))) {
			return false;
		}
	}
	if (!CudaSafeCall(cudaMallocPitch((void**)&cu_inidisp_tmp_, &idp_psize_, size_t(width_) * sizeof(sint16), size_t(height_)))) {
		return false;
	}

	inidisp_tmp_ = new sint16[width_ * height_];

	// initialize Cost Computor
	computor_ = new CostComputor;
	if(!computor_->Initialize(width_, height_, min_disparity_, min_disparity_ + disp_range_, sgm_option.cs_type)) {
		if (print_log_) printf("Initialize Cost Computor failed!\n");
		delete computor_; computor_ = nullptr;
		return false;
	}
	// initialize Cost Aggregator
	aggregator_ = new CostAggregator;
	if(!aggregator_->Initialize(width_, height_, min_disparity_, min_disparity_ + disp_range_)) {
		if (print_log_) printf("Initialize Cost Aggregator failed!\n");
		delete aggregator_; aggregator_ = nullptr;
		return false;
	}
	// initialize Disparity Filter
	filter_ = new DisparityFilter;
	if(!filter_->Initialize(width_, height_)) {
		if (print_log_) printf("Initialize Disparity Filter failed!\n");
		delete filter_; filter_ = nullptr;
		return false;
	}

	// create threads for removing peaks
	num_threads_ = std::thread::hardware_concurrency() * 3 / 4;
	remove_peaks_ = new void* [num_threads_];
	for (sint32 i = 0; i < num_threads_; i++) {
		remove_peaks_[i] = static_cast<thread_remove_peaks*>(new thread_remove_peaks);
	}
	sint32 rm_tiles = height_ / num_threads_;
	for (sint32 i = 0; i < num_threads_; i++) {
		auto thead = static_cast<thread_remove_peaks*>(remove_peaks_[i]);
		sint32 start = std::max(0, i * rm_tiles);
		sint32 end = std::min(height_, (i + 1) * rm_tiles);
		thead->w = width_;
		thead->h = end - start;
		thead->segid_ptr = new sint32[thead->w * thead->h];
		thead->threshold = thead->w * thead->h * sgm_option_.peaks_ratio_threshold;
	}

	// create concurrency streams
	cu_streams_ = static_cast<cudaStream_t*>(new cudaStream_t[2]);
	for (sint32 i = 0; i < 2; i++) {
		if (!CudaSafeCall(cudaStreamCreate(&(static_cast<cudaStream_t*>(cu_streams_))[i]))) {
			if (print_log_) printf("Create concurrency streams failed!");
			return false;
		}
	}

	if (print_log_) printf("Init Completed., Disp: %d~%d\n\n", min_disparity, disp_range + min_disparity);
	return true;
}

bool StereoCudaImpl::Init2(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option, CamParam_T cam_param, bool print_log)
{
	if (!Init(width, height, min_disparity, disp_range, sgm_option, print_log))
		return false;

	// malloc memory for disp_ptr_
#ifdef USING_PAGE_LOCKED_MEMORY
	MallocPageLockedPtr((void**)(&disp_ptr_), width * height * sizeof(float32));
#else
	disp_ptr_ = new float32[width*height];
#endif

	cam_param_ = cam_param;
	cudaMallocPitch((void**)&cu_depth_left_, &dp_psize_, size_t(width_) * sizeof(float32), size_t(height_));
	return true;
}


void StereoCudaImpl::Release()
{
	if (remove_peaks_) {
		for (sint32 i = 0; i < num_threads_; i++) {
			if (remove_peaks_[i]) {
				auto* rp = static_cast<thread_remove_peaks*>(remove_peaks_[i]);
				delete rp; remove_peaks_[i] = nullptr;
			}
		}
		delete remove_peaks_;
		remove_peaks_ = nullptr;
	}
	if (computor_) {
		computor_->Release();
		delete computor_; computor_ = nullptr;
	}
	if (aggregator_) {
		aggregator_->Release();
		delete aggregator_; aggregator_ = nullptr;
	}
	if (filter_) {
		filter_->Release();
		delete filter_; filter_ = nullptr;
	}
	SafeCudaFree(cu_img_left_);
	SafeCudaFree(cu_img_right_);
	SafeCudaFree(cu_depth_left_);
	SafeCudaFree(cu_inidisp_left_);
	SafeCudaFree(cu_inidisp_right_);
	SafeCudaFree(cu_inidisp_tmp_);
	
	if (inidisp_tmp_) {
		delete[] inidisp_tmp_;
		inidisp_tmp_ = nullptr;
	}
	
#ifdef USING_PAGE_LOCKED_MEMORY
	FreePageLockedPtr(disp_ptr_);
#else
	if(disp_ptr_) {
		delete[] disp_ptr_; disp_ptr_ = nullptr;
	}
#endif

	if (cu_streams_) {
		for (sint32 i = 0; i < 2; i++) {
			cudaStreamDestroy(static_cast<cudaStream_t*>(cu_streams_)[i]);
		}
		delete static_cast<cudaStream_t*>(cu_streams_);
		cu_streams_ = nullptr;
	}
}

float32 StereoCudaImpl::get_invad_float()
{
	return INVALID_VALUE;
}
sint16 StereoCudaImpl::get_invad_short()
{
	return INVALID_VALUE_SHORT;
}

sint16 StereoCudaImpl::get_level_range()
{
	return LEVEL_RANGE;
}

bool StereoCudaImpl::MallocPageLockedPtr(void** ptr, size_t size)
{
	const auto err = cudaHostAlloc(ptr, size, cudaHostAllocDefault);
	return err == cudaSuccess;
}

bool StereoCudaImpl::FreePageLockedPtr(void* ptr)
{
	if (ptr) {
		const auto err = cudaFreeHost(ptr);
		ptr = nullptr;
		return err == cudaSuccess;
	}
}

bool StereoCudaImpl::ComputeCost(uint8* img_left, uint8* img_right, sint16* init_disp_left/* = nullptr*/) const
{
	auto start = steady_clock::now();
	// input
	cudaStream_t stream0 = static_cast<cudaStream_t*>(cu_streams_)[0];
	cudaStream_t stream1 = static_cast<cudaStream_t*>(cu_streams_)[1];
	cudaMemcpy2DAsync(cu_img_left_, im_psize_, img_left, width_ * sizeof(uint8), width_ * sizeof(uint8), height_, cudaMemcpyHostToDevice, stream0);
	cudaMemcpy2DAsync(cu_img_right_, im_psize_, img_right, width_ * sizeof(uint8), width_ * sizeof(uint8), height_, cudaMemcpyHostToDevice, stream1);
	if (init_disp_left)
	{
		cudaMemcpy2DAsync(cu_inidisp_left_, idp_psize_, init_disp_left, width_ * sizeof(sint16), width_ * sizeof(sint16), height_, cudaMemcpyHostToDevice, stream0);
		cudaStreamSynchronize(stream0);
		if (sgm_option_.do_lr_check)
		{
			// compute initial disparity map of right view
			cusgm_util::ComputeRightInitialValue(cu_inidisp_left_, cu_inidisp_right_, width_, height_, idp_psize_);
			DisparityFilter::DilationCuda(cu_inidisp_tmp_, cu_inidisp_right_, 5, width_, height_, idp_psize_);
			cudaMemcpy2DAsync(cu_inidisp_right_, idp_psize_, cu_inidisp_tmp_, idp_psize_, idp_psize_, height_, cudaMemcpyDeviceToDevice, stream1);
		}
	}

	auto end = steady_clock::now();
	auto time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Input:				%.2lf ms\n", time);
	start = steady_clock::now();

	if (!computor_) {
		return false;
	}

	// Set data for cost computor
	computor_->SetData(cu_img_left_, cu_img_right_, im_psize_);

	// census transform for left view
	computor_->CensusTransform(true);
	cudaStreamSynchronize(stream1);

	// census transform for right view
	computor_->CensusTransform(false);

	end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Census:				%.2lf ms\n", time);
	start = steady_clock::now();

	//compute initial cost
	computor_->ComputeCost(init_disp_left ? cu_inidisp_left_ : nullptr, idp_psize_, nullptr, true);

	end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Cost:				%.2lf ms\n", time);

	return true;
}

bool StereoCudaImpl::CostAggregate(sint16* init_disp_left/* = nullptr*/, StereoROI_T* ste_roi_left/* = nullptr*/, StereoROI_T* ste_roi_right/* = nullptr*/) const
{
	auto start = steady_clock::now();

	if (!aggregator_) {
		return false;
	}

	aggregator_->SetData(cu_img_left_, cu_img_right_, im_psize_, computor_->get_cost_ptr());
	aggregator_->SetParam(sgm_option_.p1, sgm_option_.p2_init, sgm_option_.using_constant_p2, sgm_option_.unique_threshold);
	aggregator_->Aggregate(init_disp_left ? cu_inidisp_left_ : nullptr, idp_psize_, ste_roi_left);

	auto end = steady_clock::now();
	auto time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Aggregate/WTA/SubPixel:		%.2lf ms\n", time);

	start = steady_clock::now();

	if (sgm_option_.do_lr_check) {
		aggregator_->LRCheck(computor_, sgm_option_.lr_threshold, init_disp_left ? cu_inidisp_right_ : nullptr, idp_psize_, ste_roi_right);
	}

	end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** LRCheck:		%.2lf ms\n", time);


	return true;
}

void StereoCudaImpl::Filter() const
{
	auto start = steady_clock::now();

	if (filter_) {
		filter_->SetData(cu_img_left_, aggregator_->get_disp_ptr(),im_psize_, aggregator_->get_disp_psize());
		filter_->SetParam(sgm_option_.do_median_filter, sgm_option_.post_filter_type, sgm_option_.morphology_type);
		filter_->Filter();
	}

	auto end = steady_clock::now();
	auto time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Filter:		%.2lf ms\n", time);
}

void StereoCudaImpl::RemovePeaks(StereoROI_T* ste_roi_left, float32* disp_left)
{
	auto start = steady_clock::now();

	if (sgm_option_.do_remove_peaks) {
		const sint32 roi_h = (ste_roi_left == nullptr) ? height_ : ste_roi_left->h;
		const sint32 yoffset = (ste_roi_left == nullptr) ? 0 : ste_roi_left->y;
		const sint32 rm_tiles = roi_h / num_threads_;
		for (sint32 i = 0; i < num_threads_; i++) {
			sint32 start = std::max(0, yoffset + i * rm_tiles);
			sint32 end = std::min(height_, yoffset + (i + 1) * rm_tiles);
			auto* rp = static_cast<thread_remove_peaks*>(remove_peaks_[i]);
			rp->disp_ptr = disp_left + start * width_;
			rp->h = end - start;
			rp->threshold = rp->w * rp->h * sgm_option_.peaks_ratio_threshold;
			memset(rp->segid_ptr, 0, rp->w * rp->h * sizeof(sint32));
		}
		for (sint32 i = 0; i < num_threads_; i++) {
			sint32 start = std::max(0, yoffset + i * rm_tiles);
			auto* rp = static_cast<thread_remove_peaks*>(remove_peaks_[i]);
			rp->Start();
		}
		for (sint32 i = 0; i < num_threads_; i++) {
			auto* rp = static_cast<thread_remove_peaks*>(remove_peaks_[i]);
			rp->WaitForEnd();
		}
	}

	auto end = steady_clock::now();
	auto time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** RMPeaks:				%.2lf ms\n", time);

#if _TEST_TIME
	m_runtime.t_remove = time;
#endif
}

#ifdef _DEBUG
void StereoCudaImpl::OutInitValueR(sint16* pInitValue)
{
	cudaMemcpy2D(pInitValue, width_ * sizeof(sint16), cu_inidisp_right_, idp_psize_, width_ * sizeof(sint16), height_, cudaMemcpyDeviceToHost);
}
#endif

bool StereoCudaImpl::Match(uint8* img_left, uint8* img_right, float32* disp_left, float32* disp_right, sint16* init_disp_left/* = nullptr*/, StereoROI_T* ste_roi_left/* = nullptr*/, StereoROI_T* ste_roi_right/* = nullptr*/)
{
	if (img_left == nullptr || img_right == nullptr) {
		return false;
	}

	float64 time = 0.0;
	if (print_log_) printf("Timing:\n");

	// compute cost
	if (!ComputeCost(img_left, img_right, init_disp_left)) {
		return false;
	}

	// cost aggregate
	if (!CostAggregate(init_disp_left, ste_roi_left, ste_roi_right)) {
		return false;
	}
	dp_psize_ = aggregator_->get_disp_psize();

	// filter
	Filter();

	auto start = steady_clock::now();

	// output disparities to host
	cudaError_t status = cudaMemcpy2D(disp_left, width_ * sizeof(float32), filter_->get_disp_map_out(), dp_psize_, width_ * sizeof(float32), height_, cudaMemcpyDeviceToHost);
	if (disp_right) {
		cudaMemcpy2D(disp_right, width_ * sizeof(float32), aggregator_->get_disp_r_ptr(), dp_psize_, width_ * sizeof(float32), height_, cudaMemcpyDeviceToHost);
	}

	auto end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** D2H:					%.2lf ms\n", time);
	
	// remove peaks
	RemovePeaks(ste_roi_left, disp_left);

	return true;
}

bool StereoCudaImpl::Match2(uint8* img_left, uint8* img_right, float32* depth_left, sint16* init_disp_left/* = nullptr*/, StereoROI_T* ste_roi_left/* = nullptr*/, StereoROI_T* ste_roi_right/* = nullptr*/)
{
	if (img_left == nullptr || img_right == nullptr || disp_ptr_ == nullptr) {
		return false;
	}

	float64 time = 0.0, total_time = 0.0;
	auto start = steady_clock::now();

	Match(img_left, img_right, disp_ptr_, nullptr, init_disp_left, ste_roi_left, ste_roi_right);

	auto end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;

	start = steady_clock::now();

	// compute depth
	if (sgm_option_.do_remove_peaks)
		cudaMemcpy2D(aggregator_->get_disp_ptr(), dp_psize_, disp_ptr_, width_ * sizeof(float32), width_ * sizeof(float32), height_, cudaMemcpyHostToDevice);
	cusgm_util::ComputeDepthCuda(aggregator_->get_disp_ptr(), dp_psize_, width_, height_, cu_depth_left_, cam_param_);

	end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** DepthComp:				%.2lf ms\n", time);

	start = steady_clock::now();

	// output depth to host
	cudaMemcpy2D(depth_left, width_ * sizeof(float32), cu_depth_left_, dp_psize_, width_ * sizeof(float32), height_, cudaMemcpyDeviceToHost);

	end = steady_clock::now();
	time = (duration_cast<microseconds>(end - start)).count() / 1000.0;
	if (print_log_) printf("** Output:				%.2lf ms\n", time);

	return true;
}

void StereoCudaImpl::GetRoiFromDispMap(float32* disp_ptr, sint32 width, sint32 height, StereoROI_T& ste_roi)
{
	if (!disp_ptr) {
		return;
	}
	sint32 size = width * height;
	float32* disp = disp_ptr;
	sint32 n = 0;
	for (n = 0; n < size; n++) {
		if (*(disp++) != INVALID_VALUE) {
			break;
		}
	}
	ste_roi.y = n / width;
	disp = disp_ptr + size - 1;
	sint32 size0 = ste_roi.y * width;
	for (n = size - 1; n > size0; n--)
		if (*(disp--) != INVALID_VALUE)
			break;
	ste_roi.h = n / width - ste_roi.y + 1;

	bool search_done = false;
	for (sint32 j = 0; j < width; j++) {
		for (sint32 i = 0; i < height; i++) {
			if (disp_ptr[i * width + j] != INVALID_VALUE) {
				ste_roi.x = j;
				search_done = true;
				break;
			}
		}
		if (search_done) break;
	}
	search_done = false;
	for (sint32 j = width - 1; j > ste_roi.x; j--) {
		for (sint32 i = 0; i < height; i++) {
			if (disp_ptr[i * width + j] != INVALID_VALUE) {
				ste_roi.w = j - ste_roi.x + 1;
				search_done = true;
				break;
			}
		}
		if (search_done) break;
	}
}

void StereoCudaImpl::SetMinDisparity(const sint32& min_disparity)
{
	min_disparity_ = min_disparity;
	computor_->SetMinDisparity(min_disparity);
	aggregator_->SetMinDisparity(min_disparity);
}

sint32 StereoCudaImpl::GetDispartyRange()
{
	return disp_range_;
}
