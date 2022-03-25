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

//��ʼ���㷨���
ErrAlgorithm AlgorithmPlugin::pluginInitialize(const PluginParam& pluginParam)
{
	m_inited = true;
	GlobalParm::instance().loadConfig(pluginParam);
	AlgoMsgDebug(m_logger, "MultiTargetDetection init success!");
	return ErrALGOSuccess; 
}

//�ͷ��㷨���
ErrAlgorithm AlgorithmPlugin::pluginRelease()
{
	if (!m_inited)
	{
		return ErrALGOSuccess;
	}
	AlgoMsgDebug(m_logger, "MultiTargetDetection release success!");
	return ErrALGOSuccess;
}

//����������Video Analysis�㷨
shared_ptr<AlgorithmVAInterface> AlgorithmPlugin::createVAAlgorithm()
{	
	if (!m_inited)
	{
		return nullptr;
	}
	return std::make_shared<VideoAnalysis>();
}

void AlgorithmPlugin::destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo)
{
	return ;
}

//����������Image Retrieval�㷨
shared_ptr<AlgorithmIRInterface> AlgorithmPlugin::createIRAlgorithm()
{
	if (!m_inited)
	{
		return nullptr;
	}
	return std::make_shared<ImageRetrieval>();;
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