#pragma once
#include "windows.h"

class MyTimer
{
public:
	MyTimer() { QueryPerformanceFrequency(&m_litF); }
	~MyTimer() {}

	inline void Start();
	inline void End();
	inline double GetDuration() { return  m_duration; }
	inline double GetDurationMs() { return m_duration * 1000; }
	inline int GetDurHH() { return m_HH; }
	inline int GetDurMM() { return m_MM; }
	inline int GetDurSS() { return m_SS; }
private:
	LARGE_INTEGER m_litS, m_litE, m_litF;
	double m_duration;
	int m_HH, m_MM, m_SS;
};

inline void MyTimer::Start()
{
	QueryPerformanceCounter(&m_litS);
}

inline void MyTimer::End()
{
	QueryPerformanceCounter(&m_litE);
	m_duration = (m_litE.QuadPart*1.0 - m_litS.QuadPart) / (m_litF.QuadPart);
	m_HH = (int)m_duration / 3600;
	m_MM = ((int)m_duration - m_HH * 3600) / 60;
	m_SS = (int)m_duration - m_HH * 3600 - m_MM * 60;
}