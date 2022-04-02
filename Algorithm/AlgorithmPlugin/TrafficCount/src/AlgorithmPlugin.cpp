#include "AlgorithmPlugin.h"
#include "Poco/ClassLibrary.h"
#include <iostream>
#include <utility>
#include <mutex>
#include <thread>
#include <chrono>
#include "GlobalParm.h"
#include "VideoAnalysis.h"
#include "ImageRetrieval.h"

#if !defined(_WIN32)
#include<pthread.h>
#endif

AlgorithmPlugin::AlgorithmPlugin()
{

}

AlgorithmPlugin::~AlgorithmPlugin()
{

}

//初始化算法插件
ErrAlgorithm AlgorithmPlugin::pluginInitialize(const PluginParam& pluginParam)
{
	m_inited = true;
	GlobalParm::instance().loadConfig(pluginParam);
	AlgoMsgDebug(m_logger, "MultiTargetDetection init success!");
	return ErrALGOSuccess; 
}

//释放算法插件
ErrAlgorithm AlgorithmPlugin::pluginRelease()
{
	if (!m_inited)
	{
		return ErrALGOSuccess;
	}
	AlgoMsgDebug(m_logger, "MultiTargetDetection release success!");
	return ErrALGOSuccess;
}

//创建、销毁Video Analysis算法
shared_ptr<AlgorithmVAInterface> AlgorithmPlugin::createVAAlgorithm(int gpuId)
{	
	if (!m_inited)
	{
		return nullptr;
	}

	return std::make_shared<VideoAnalysis>(gpuId);
}

void AlgorithmPlugin::destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo)
{
	return ;
}

//创建、销毁Image Retrieval算法
shared_ptr<AlgorithmIRInterface> AlgorithmPlugin::createIRAlgorithm(int gpuId)
{
	if (!m_inited)
	{
		return nullptr;
	}

	return std::make_shared<ImageRetrieval>(gpuId);;
}

void AlgorithmPlugin::destoryIRAlgorithm(shared_ptr<AlgorithmIRInterface> algo)
{
	return;
}

POCO_BEGIN_MANIFEST(AlgorithmPluginInterface)
POCO_EXPORT_CLASS(AlgorithmPlugin)
POCO_END_MANIFEST


void pocoInitializeLibrary()
{
	std::cout << "PluginDemo initializing" << std::endl;
}


void pocoUninitializeLibrary()
{
	std::cout << "PluginDemo uninitialzing" << std::endl;
}