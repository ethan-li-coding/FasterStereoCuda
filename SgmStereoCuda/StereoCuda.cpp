#include "StereoCuda.h"
#include "StereoCudaImpl.h"

StereoCuda::StereoCuda(): impl_(nullptr)
{
	impl_ = new StereoCudaImpl;
}

StereoCuda::~StereoCuda()
{
	if(impl_) {
		auto impl = static_cast<StereoCudaImpl*>(impl_);
		delete impl;
		impl = nullptr;
	}
}

bool StereoCuda::Init(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option,
	bool print_log) const
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->Init(width, height, min_disparity, disp_range, sgm_option, print_log);
	}
	else {
		return false;
	}
}

bool StereoCuda::Match(uint8* img_left, uint8* img_right, float32* disp_left, float32* disp_right,
	sint16* init_disp_left, StereoROI_T* ste_roi_left, StereoROI_T* ste_roi_right) const
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->Match(img_left, img_right, disp_left, disp_right, init_disp_left, ste_roi_left, ste_roi_right);
	}
	else {
		return false;
	}
}

bool StereoCuda::Init2(sint32 width, sint32 height, sint32 min_disparity, sint32 disp_range, CuSGMOption sgm_option,
	CamParam_T cam_param, bool print_log) const
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->Init2(width, height, min_disparity, disp_range, sgm_option,cam_param, print_log);
	}
	else {
		return false;
	}
}

bool StereoCuda::Match2(uint8* img_left, uint8* img_right, float32* depth_left, sint16* init_disp_left,
	StereoROI_T* ste_roi_left, StereoROI_T* ste_roi_right) const
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->Match2(img_left, img_right, depth_left,  init_disp_left, ste_roi_left, ste_roi_right);
	}
	else {
		return false;
	}
}

void StereoCuda::SetMinDisparity(const sint32& min_disparty)
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->SetMinDisparity(min_disparty);
	}
}

sint32 StereoCuda::GetDispartyRange()
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		return impl->GetDispartyRange();
	}
}

void StereoCuda::GetRoiFromDispMap(float32* disp_ptr, sint32 width, sint32 height, StereoROI_T& ste_roi)
{
	StereoCudaImpl::GetRoiFromDispMap(disp_ptr, width, height, ste_roi);
}

void StereoCuda::Release() const
{
	auto impl = static_cast<StereoCudaImpl*>(impl_);
	if (impl) {
		impl->Release();
	}
}

float32 StereoCuda::get_invad_float()
{
	return StereoCudaImpl::get_invad_float();
}

sint16 StereoCuda::get_invad_short()
{
	return StereoCudaImpl::get_invad_short();
}

sint16 StereoCuda::get_level_range()
{
	return StereoCudaImpl::get_level_range();
}

bool StereoCuda::MallocPageLockedPtr(void** ptr, size_t size)
{
	return StereoCudaImpl::MallocPageLockedPtr(ptr, size);
}

bool StereoCuda::FreePageLockedPtr(void* ptr)
{
	return StereoCudaImpl::FreePageLockedPtr(ptr);
}
