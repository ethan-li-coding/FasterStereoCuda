#include <opencv.hpp>
#include "SgmStereoCuda.h"
#include <string>

using namespace cv;
using namespace std;

#ifdef _DEBUG
#pragma comment(lib,"opencv_world320d.lib")
#else
#pragma comment(lib,"opencv_world320.lib")
#endif



int main(int argc, char** argv)
{
	if (argc<3)
	{
		argv = new char*[3];
		argv[1] = "Y:\\---项目---\\ZG-白光扫描仪\\程序\\PySgmTest\\x64\\Release\\IMG0_L.BMP";
		argv[2] = "Y:\\---项目---\\ZG-白光扫描仪\\程序\\PySgmTest\\x64\\Release\\IMG0_R.BMP";
	}

	//读取影像
	Mat leftImg = imread(argv[1], IMREAD_GRAYSCALE);
	Mat rightImg = imread(argv[2], IMREAD_GRAYSCALE);
	if (leftImg.rows != rightImg.rows || leftImg.cols != rightImg.cols)
	{
		return -1;
	}
	//宽-高
	int width = leftImg.cols;
	int height = leftImg.rows;

	unsigned char* pLeft = new unsigned char[width*height]
		, *pRight = new unsigned char[width*height];
	float *pDisp = new float[width*height];
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			pLeft[i*width + j] = leftImg.data[(height - 1 - i)*width + j];
			pRight[i*width + j] = rightImg.data[(height - 1 - i)*width + j];
		}
	}

	//视差图
	string dispImgPath = argv[1];
	string tmpAppend = ".disp.bmp";
	dispImgPath.append(tmpAppend);
	Mat dispImg16(leftImg.rows, leftImg.cols, CV_16SC1);
	Mat dispImg32(leftImg.rows, leftImg.cols, CV_32FC1);

	SgmStereoCuda sgmCuda;
	SgmParam_T sgmt;
	sgmt.bAggregating = true;
	sgmCuda.Init(width, height, 196, 128, sgmt);
	sgmCuda.Match(pLeft, pRight, pDisp);
	sgmCuda.Release();

	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			dispImg32.ptr<float>(0)[i*width + j] = pDisp[(height - 1 - i)*width + j];
		}
	}
	dispImgPath = argv[1];
	tmpAppend = ".disp-SC.bmp";
	dispImgPath.append(tmpAppend);
	imwrite(dispImgPath, dispImg32);
}