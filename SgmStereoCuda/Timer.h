#pragma once
#include <chrono>
using namespace std::chrono;

class MyTimer
{
public:
	MyTimer() {}
	~MyTimer() {}

	inline void Start();
	inline void End();
	double GetDurationS() { return  duration_; }
	double GetDurationMS() { return duration_ * 1000; }
	int GetDurH() { return hours_; }
	int GetDurM() { return minutes_; }
	int GetDurS() { return seconds_; }
private:
	steady_clock::time_point start_, end_;
	double duration_;
	int hours_, minutes_, seconds_;
};

inline void MyTimer::Start()
{
	start_ = steady_clock::now();
}

inline void MyTimer::End()
{
	end_ = steady_clock::now();
	duration_ = duration_cast<seconds>(end_ - start_).count();
	hours_ = duration_ / 3600;
	minutes_ = (duration_ - hours_ * 3600) / 60;
	seconds_ = duration_ - hours_ * 3600 - minutes_ * 60;
}