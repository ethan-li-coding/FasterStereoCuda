#pragma once
#include <chrono>
#include <corecrt_io.h>
#include <errno.h>
#include <string>
#include <iostream>


/**
 * \brief 试用期检查
 */
class TrialCheck {
public:
	TrialCheck();
	~TrialCheck();

	/**
	 * \brief 检查器
	 */
	static inline int Check(const int& trial_period = 180) {
		const std::string trial_name = "./fasterstereocuda.a";

		typedef std::chrono::duration<int, std::ratio<60 * 60 * 24>> days_type;
		std::chrono::time_point<std::chrono::system_clock, days_type> today = std::chrono::time_point_cast<days_type>(std::chrono::system_clock::now());
		int time_cur = today.time_since_epoch().count();
		double time_cur_public = time_cur * exp(1);
		double time_last_public = 0.0, time_first_public = 0.0;

		if(_access(trial_name.c_str(),0) == -1) {
			std::string err_code;
			if (errno == ENOENT) {
				err_code = "No File";
			}
			else if (errno == EACCES) {
				// 没有访问权限
				err_code = "Access Ddenied";
			}
			printf("License Not Found! [%s]", err_code.c_str());
			return -1;
		}

		FILE* fp_trail = fopen(trial_name.c_str(), "rb");

		if(!fp_trail) {
			printf("License Cant't be Opened!");
			return -1;
		}

		int time_last = 0, time_first = 0;

		fread(&time_last_public, sizeof(double), 1, fp_trail);
		fread(&time_first_public, sizeof(double), 1, fp_trail);

		time_last = int(time_last_public / exp(1));
		time_first = int(time_first_public / exp(1));

		if(time_last > time_cur) {
			printf("License Time Error!");
			return -1;
		}

		// 试用期n个月
		if (time_cur - time_first > trial_period) {
			printf("License Time Exceed!");
			return -1;
		}

		fwrite(&time_cur_public, sizeof(double), 1, fp_trail);
		fwrite(&time_first_public, sizeof(double), 1, fp_trail);
		return trial_period - time_cur + time_first;
	}

	/**
	 * \brief 生成器
	 */
	static inline void Generate() {
		const std::string trial_name = "fasterstereocuda.a";

		typedef std::chrono::duration<int, std::ratio<60 * 60 * 24>> days_type;
		std::chrono::time_point<std::chrono::system_clock, days_type> today = std::chrono::time_point_cast<days_type>(std::chrono::system_clock::now());
		int time_cur = today.time_since_epoch().count();
		double time_cur_public = time_cur * exp(1);
		double time_last_public = 0.0, time_first_public = 0.0;

		FILE* fp_trail = fopen(trial_name.c_str(), "wb");
		time_first_public = time_cur_public;
		fwrite(&time_cur_public, sizeof(double), 1, fp_trail);
		fwrite(&time_first_public, sizeof(double), 1, fp_trail);
		fclose(fp_trail);
	}
};
