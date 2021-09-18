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
	//����������,һ�������Ķ�Ӧһ����������
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

	//tag ��ͬ��tag��ʹ�ò�ͬ��cuda context��������
	virtual void inference(float* input, float* output, int batchSize, const std::string tag = "default", bool allocMaxBatch = false);

	virtual void getDDims(DDIms&);

	virtual void destroy();

protected:
	virtual string getClassName() = 0;

	//��ʼ����������
	virtual bool initializeEngine();

	//���л������ļ�������
	virtual void serializeEngine();

	//�Ӵ��̷����л������ļ����ڴ�
	virtual bool deserializeEngine();

	//��ʼ��������Ҫ�������ļ�Input Data, Output Data
	virtual bool initCudaResource(string tag, int batchSize);

	//����ʱ�䲻���õ�Context
	virtual void checkOuttimeContext();

	//ģ�͹���,���������д
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

