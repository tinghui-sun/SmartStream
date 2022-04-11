#pragma once
#include "Algorithm.h"
#include <map>
#include <atomic>
#include <thread>

class AlgorithmPlugin : public AlgorithmPluginInterface
{
public:
	AlgorithmPlugin();

	virtual ~AlgorithmPlugin();

	//初始化算法插件
	virtual ErrAlgorithm pluginInitialize(const PluginParam& pluginParam, int gpuId) override;

	//释放算法插件
	virtual ErrAlgorithm pluginRelease() override;

	//创建、销毁Video Analysis算法
	virtual shared_ptr<AlgorithmVAInterface> createVAAlgorithm(int gpuId) override;
	virtual void destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo) override;

	//创建、销毁Image Retrieval算法
	virtual shared_ptr<AlgorithmIRInterface> createIRAlgorithm(int gpuId) override;
	virtual void destoryIRAlgorithm(shared_ptr<AlgorithmIRInterface> algo) override;

private:
	PluginParam m_pluginParam;
	shared_ptr<AlgoLoggerInterface> m_logger;
	bool m_inited = false;
};