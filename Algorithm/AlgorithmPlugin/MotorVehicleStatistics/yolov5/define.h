#pragma once
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
//#include "cudefine.h"
using namespace std;

struct AutoDeleter
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			obj->destroy();
		}
	}
};

struct AutoDeleter1
{
	template <typename T>
	void operator()(T* obj) const
	{
		if (obj)
		{
			delete[]obj;
			obj = nullptr;
		}
	}
};


template <typename T>
using UniquePtr = unique_ptr<T, AutoDeleter>;

//template <typename T>
//using SharedPtr = shared_ptr<T>T, AutoDeleter());

template <typename T>
using UniquePtr1 = unique_ptr<T, AutoDeleter1>;

//template <typename T>
//using SharedPtr1 = shared_ptr<T>(T, AutoDeleter1());

struct ModelParams 
{
	int batchSize{ 0 };              
	int dlaCore{ -1 };                   
	bool int8{ false };    
	float confThresh{ 0.5f };
	float nmsThresh{ 0.4f };
	string dataDir; 
	string weightsFileName;
	string engineFileName;
	string calibrationFileName;
	string calibrationImgFolder;
	int calibrationImgBatchs;
	string modeType; //  [s/m/l/x/s6/m6/l6/x6 or c/c6 gd gw]

	string to_string() {
		return "\n  batchSize:" + std::to_string(batchSize) + "\n  dlaCore:" + std::to_string(dlaCore) + "\n  int8:" + std::to_string(int8) + "\n  confThresh:" + std::to_string(confThresh) + "\n  nmsThresh:" + std::to_string(nmsThresh) + "\n  weightsFileName:" + weightsFileName
			+ "\n  engineFileName:" + engineFileName + "\n  calibrationFileName:" + calibrationFileName + "\n  calibrationImgFolder:" + calibrationImgFolder + "\n  modeType:" + modeType + "\n  dataDir:" + dataDir;
	}
};

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret  << " at " << __FILE__ << ":" << __LINE__<< std::endl;\
            abort();\
        }\
    } while (0)



struct Yolov5Detect
{
	int classId;
	string className;
	float confidence;
	cv::Rect rect;
	cv::Mat img;
};

typedef struct TrackingBox
{
	int id;  //目标索引
	cv::Rect objTrackRect;//跟踪输出的坐标
	cv::Rect objRect;//当前轮跟踪时输入的坐标
}TrackingBox;

//return false退出文件夹读取
using Yolov5InferDirCallback = bool(*)(vector<string>* files, vector<vector<Yolov5Detect>>* result);

//return false退出流读取
using Yolov5InferStreamCallback = bool(*)(cv::Mat* frame);