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
#include "VimoDetectionModule.h"

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
ErrAlgorithm AlgorithmPlugin::pluginInitialize(const PluginParam& pluginParam, int gpuId)
{
	if(!m_inited)
	{
		if(!GlobalParm::instance().loadConfig(pluginParam))
		{
			AlgoMsgError(m_logger, "LeaveDetection init failed!");
			return ErrALGOConfigInvalid;
		}
		else
		{
		  	//Engine
  			smartmore::VimoDetectionModule engine;
			smartmore::ResultCode rs = engine.Init(GlobalParm::instance().m_modlePath, false, gpuId);
			if(rs != smartmore::ResultCode::Success)
			{
				AlgoMsgError(m_logger, "LeaveDetection.init failed![%s]", GlobalParm::instance().m_modlePath.c_str());
				return ErrALGOInitFailed;
			}
		}
  		m_inited = true;
	}

	AlgoMsgDebug(m_logger, "LeaveDetection init success!");
	return ErrALGOSuccess; 
}

//释放算法插件
ErrAlgorithm AlgorithmPlugin::pluginRelease()
{
	if (!m_inited)
	{
		return ErrALGOSuccess;
	}
	AlgoMsgDebug(m_logger, "LeaveDetection release success!");
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
	std::cout << "LeaveDetection initializing" << std::endl;
}


void pocoUninitializeLibrary()
{
	std::cout << "LeaveDetection uninitialzing" << std::endl;
}