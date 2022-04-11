#pragma once
#include "Algorithm.h"
#include "VimoDetectionModule.h"

class VideoAnalysis: public AlgorithmVAInterface
{
public:
	VideoAnalysis(int gpuId);
	virtual ~VideoAnalysis();

public:
	virtual ALGOAbility getAbility() override;
	virtual ErrAlgorithm analyzeImageASync(const std::list <ALGOImageInfo>& imageList) override;
	virtual ErrAlgorithm analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResults) override;

private:
	ErrAlgorithm initEngine();
	
private:
  //Engine
  smartmore::VimoDetectionModule mEngine;
  
  //Tracker
  bool inited = false;
  int mGPUId = 0;
};

