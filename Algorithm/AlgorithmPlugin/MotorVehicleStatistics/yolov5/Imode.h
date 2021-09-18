#pragma once
#include <string>

struct DDIms
{
	int width{ 0 };
	int height{ 0 };
	int channel{ 0 };
	int batchSize{ 0 };
	int outSize{ 0 };
	float confThresh{0.5f};
	float nmsThresh{ 0.4f };
};

class IMode
{
public:
	virtual bool initialize(const char* configPath) = 0;
	virtual void inference(float* input, float* output, int batchSize, const std::string tag = "default",  bool allocMaxBatch = false) = 0;
	virtual void getDDims(DDIms&) = 0;
	virtual void destroy() = 0;
};

