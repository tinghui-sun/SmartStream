#pragma once
#include "Algorithm.h"
class VideoAnalysis: public AlgorithmVAInterface
{
public:
	VideoAnalysis(int gpuId);
	virtual ~VideoAnalysis();

public:
	virtual ALGOAbility getAbility() override;
	virtual ErrAlgorithm analyzeImageASync(const std::list <ALGOImageInfo>& imageList) override;
	virtual ErrAlgorithm analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResult) override;
};

