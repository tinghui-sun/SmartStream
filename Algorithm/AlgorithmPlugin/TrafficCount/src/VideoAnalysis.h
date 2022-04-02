#pragma once
#include "Algorithm.h"
#include "ByteTrack/BYTETracker.h"
#include "TrafficCounter.h"

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
  std::vector<std::string> mClassesNames;
  //Engine
  smartmore::TrafficCounter mTrafficCountingEngine;

  //Tracker
  std::vector<byte_track::BYTETracker> mTrackers;
  bool inited = false;
  int mGPUId = 0;
};

