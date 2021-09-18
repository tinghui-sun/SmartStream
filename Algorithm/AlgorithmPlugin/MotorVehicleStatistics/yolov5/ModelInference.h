#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include "logger.h"
#include "define.h"
#include "imode.h"
#include "utils.h"

using namespace nvinfer1;
using namespace std;

enum StreamType
{
	STFILE,
};

class ModelInference
{
public:
	ModelInference();
	virtual ~ModelInference();

public:
	bool initialize(string configPath = "", int gpuId = 0);
	bool run(string imgDir, Yolov5InferDirCallback callback, string tag = "default");
	bool run(string file, vector<Yolov5Detect>& results, string tag = "default");
	bool run(cv::Mat img, vector<Yolov5Detect>& results, string tag = "default");
	//加个错误码吧 1 图片异常 2 图片转换失败  0 成功
	bool run(vector<pair<string, cv::Mat>>& imgs, vector<pair<int, vector<Yolov5Detect>>>& results,  vector<int> &classFilter, pair<int,int> sizeFilter = make_pair(60, 60), string tag = "default");
	bool runStream(string url, StreamType type, Yolov5InferStreamCallback callback, int fps = 25, string tag = "default");
	void runTest();
	void bindLogWriter(std::function<void(int, const char*)> writer)
	{
		Logger::GetInstance()->bindLogWriter(std::forward<std::function<void(int, const char*)>>(writer));
	}

private:
	void parserYoloResult(vector<float>& data, vector<vector<Yolo::Detection>>& result);
private:
	UniquePtr<IMode> m_mode;
	map<int, string> m_className;
};

