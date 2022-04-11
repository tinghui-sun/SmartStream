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

	//��ʼ���㷨���
	virtual ErrAlgorithm pluginInitialize(const PluginParam& pluginParam, int gpuId) override;

	//�ͷ��㷨���
	virtual ErrAlgorithm pluginRelease() override;

	//����������Video Analysis�㷨
	virtual shared_ptr<AlgorithmVAInterface> createVAAlgorithm(int gpuId) override;
	virtual void destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo) override;

	//����������Image Retrieval�㷨
	virtual shared_ptr<AlgorithmIRInterface> createIRAlgorithm(int gpuId) override;
	virtual void destoryIRAlgorithm(shared_ptr<AlgorithmIRInterface> algo) override;

private:
	PluginParam m_pluginParam;
	shared_ptr<AlgoLoggerInterface> m_logger;
	bool m_inited = false;
};