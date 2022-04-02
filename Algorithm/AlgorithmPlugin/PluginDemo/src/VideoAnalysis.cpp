#include "VideoAnalysis.h"
#include "GlobalParm.h"

VideoAnalysis::VideoAnalysis(int gpuId)
{
}


VideoAnalysis::~VideoAnalysis()
{
}

ALGOAbility VideoAnalysis::getAbility()
{
	ALGOAbility ability;
	ability.dataType = ALGOInterfaceParamType_JPEG_STRING;
	return ability;
}

ErrAlgorithm VideoAnalysis::analyzeImageASync(const std::list <ALGOImageInfo>& imageList)
{

	auto strongListener = m_vaResListener.lock();
	if (!strongListener)
	{
		AlgoMsgError(GlobalParm::instance().m_logger, "registerAVResListener first!");
		return ErrALGOUnSupport;
	}

	AlgoMsgDebug(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImageASync start!");

	//TODO 启动独立分析线程执行分析任务
	std::list <ALGOVAResult> vaResult;
	ALGOVAResult result;
	result.code = ErrALGOSuccess;
	result.imageInfo;
	result.objParams;
	result.statisticsNum; 

	vaResult.emplace_back(result);


	strongListener->algorithmVAFinished(vaResult);

	//return ErrALGOUnSupport;
	return ErrALGOSuccess;
}

ErrAlgorithm VideoAnalysis::analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResult)
{
	AlgoMsgDebug(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage start!");

	vaResult.clear();
	ALGOVAResult result;
	result.code = ErrALGOSuccess;
	result.imageInfo;
	result.objParams;
	result.statisticsNum; 

	vaResult.emplace_back(result);

	//return ErrALGOUnSupport;
	return ErrALGOSuccess;
}