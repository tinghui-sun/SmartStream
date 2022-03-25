#pragma once
#include "Algorithm.h"
class VideoAnalysis: public AlgorithmVAInterface
{
public:
	VideoAnalysis();
	virtual ~VideoAnalysis();

public:
	virtual ALGOAbility getAbility() override;
	virtual ErrAlgorithm analyzeImage(const std::list <ALGOImageInfo>& imageList) override;
};

