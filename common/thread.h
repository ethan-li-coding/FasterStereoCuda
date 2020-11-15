#ifndef _THREAD_BASE_H_
#define _THREAD_BASE_H_
#include "semaphore.h"

// thread 
typedef class ThreadBase {
public:
	int thread_id;			// thread id
	Semaphore* sem_start{};
	Semaphore* sem_end{};
	bool running;			// running
	bool waiting;			// waiting
	ThreadBase() {
		thread_id = 0;
		sem_start = new Semaphore;
		sem_end = new Semaphore;
		running = true;
		waiting = false;
	}
	// Wait to start
	void WaitToStart() {
		waiting = true;
		sem_start->Wait();
		waiting = false;
	}
	// Wait for ending
	void WaitForEnd() const{ sem_end->Wait(); }
	// Start
	void Start() const { sem_start->Signal(); }
	// End
	void End() const { sem_end->Signal(); }
	// Wait for terminating
	void WaitForTerminate() {
		running = false;
		if (waiting) { Start(); }
		sem_end->Wait();
		if(sem_start){ delete sem_start; sem_start = nullptr; }
		if(sem_end){ delete sem_end; sem_end = nullptr; }
	}
}thread_base;

#endif // _THREAD_BASE_H_