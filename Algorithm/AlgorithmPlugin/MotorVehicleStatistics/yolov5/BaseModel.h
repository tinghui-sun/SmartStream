#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <vector>
#include <map>
#include <mutex>
#include "IMode.h"
#include "Logger.h"
#include "define.h"
#include "TinyTimer.h"

using namespace nvinfer1;
using namespace std;

const static string INPUT_BLOB_NAME = "input";
const static string OUTPUT_BLOB_NAME = "output";

class BaseModel : public IMode
{
protected:
	//推理上下文,一个上下文对应一个推理任务
	struct EngineContext
	{
		IExecutionContext* context{ nullptr };
		cudaStream_t cudaStream{ nullptr };
		vector<void*> cudaBuffers;
		int inputBindingIndex{ -1 };
		int outputBindingIndex{ -1 };
		int batchSize{ -1 };
		long long lastTime{ 0 };

		virtual ~EngineContext()
		{
			context ? context->destroy() : void();
			context = nullptr;
			if (cudaStream)
				CHECK(cudaStreamDestroy(cudaStream));
			cudaStream = nullptr;
			for (int i = 0; i < (int)cudaBuffers.size(); i++)
			{
				CHECK(cudaFree(cudaBuffers[i]));
				cudaBuffers[i] = nullptr;
			}
		}
	};

public:
	BaseModel(int inputWidth, int inputHeight, int inputChannels, int outputSize);
	virtual ~BaseModel();

public:	
	virtual bool initialize(const char* configPath);

	//tag 不同的tag会使用不同的cuda context进行推理。
	virtual void inference(float* input, float* output, int batchSize, const std::string tag = "default", bool allocMaxBatch = false);

	virtual void getDDims(DDIms&);

	virtual void destroy();

protected:
	virtual string getClassName() = 0;

	//初始化推理引擎
	virtual bool initializeEngine();

	//序列化引擎文件到磁盘
	virtual void serializeEngine();

	//从磁盘反序列化引擎文件到内存
	virtual bool deserializeEngine();

	//初始化推理需要的上下文及Input Data, Output Data
	virtual bool initCudaResource(string tag, int batchSize);

	//清理长时间不适用的Context
	virtual void checkOuttimeContext();

	//模型构造,子类必须重写
	virtual bool createNetwork(INetworkDefinition* network, map<string, Weights>& weightMap) = 0;

protected:
	ModelParams m_params;
	UniquePtr<IBuilder> m_builder;
	UniquePtr<IRuntime> m_runtime;
	UniquePtr<ICudaEngine> m_engine;
	unique_ptr<IInt8Calibrator> m_int8calibrator;
	mutex m_engineContextsMtu;
	map<string, shared_ptr<EngineContext>> m_engineContexts;	
	shared_ptr<TinyTimer> m_contextCleanTimer;	

	int m_inputWidht{ 0 };
	int m_inputHight{ 0 };
	int m_inputChannels{ 0 };
	int m_outputSize{ 0 };
	int m_maxBatchSize{ 0 };
};

