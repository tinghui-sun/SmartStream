#include "VideoAnalysis.h"
#include "GlobalParm.h"

VideoAnalysis::VideoAnalysis()
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

ErrAlgorithm VideoAnalysis::analyzeImage(const std::list <ALGOImageInfo>& imageList)
{
	AlgoMsgDebug(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage start!");

	std::list <ALGOVAResult> vaResult;
	ALGOVAResult result;
	result.code = ErrALGOSuccess;
	result.imageInfo;
	result.objParams;
	result.statisticsNum; 

	vaResult.emplace_back(result);

	auto strongListener = m_vaResListener.lock();
	if (strongListener)
	{
		strongListener->algorithmVAFinished(vaResult);
	}
	return ErrALGOSuccess;
}