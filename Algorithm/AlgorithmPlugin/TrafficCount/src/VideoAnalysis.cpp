#include "VideoAnalysis.h"
#include "GlobalParm.h"
#include "Utils.h"
#include "cuda_runtime_api.h"
#include "cuda.h"

#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

VideoAnalysis::VideoAnalysis(int gpuId):mGPUId(gpuId)
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
	return ErrALGOUnSupport;
}

ErrAlgorithm VideoAnalysis::analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResults)
{
	AlgoMsgDebug(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage start!");

	ErrAlgorithm rs = ErrALGOSuccess;
	if((rs = initEngine()) != ErrALGOSuccess)
	{
		return rs;
	}

	if(imageList.empty())
	{
		AlgoMsgError(GlobalParm::instance().m_logger, "imageList is empty!");
		return ErrALGOParamInvalid;
	}

	vaResults.clear();

	for(auto& imageInfo:imageList)
	{
		smartmore::TrafficCounterRequest trafficCounterReq;
		smartmore::TrafficCounterResponse trafficCounterRsp;

		//cv::Mat image = cv::imread(imageFolderPath + "/" + imagePath);
		cv::Mat image;
		if(imageInfo.imageBufferType == ALGOBufferGPU)
		{
			AlgoMsgError(GlobalParm::instance().m_logger, "gpu frame");
			image = cv::Mat(imageInfo.imageHeight, imageInfo.imageWidth, CV_8UC3);
			cudaSetDevice(mGPUId);		
			auto cudaErr = cudaMemcpy(image.data, imageInfo.imageBuffer, imageInfo.imageWidth * imageInfo.imageHeight * 3, cudaMemcpyDeviceToHost);	
		}
		else
		{
			AlgoMsgError(GlobalParm::instance().m_logger, "cpu frame");
			if(imageInfo.imageFormate == ALGOImageFormatCVMat)
			{
				image = cv::Mat(imageInfo.imageHeight, imageInfo.imageWidth, CV_8UC3, (uchar*)imageInfo.imageBuffer);
			}
			else if(imageInfo.imageFormate == ALGOImageFormatJpeg)
			{
				vector<char> buf;
				for(int i = 0; i <imageInfo.imageBufferLen; i++)
					buf.emplace_back(imageInfo.imageBuffer[i]);
				image = cv::imdecode(buf, cv::IMREAD_COLOR);
			}
			else
			{
				return ErrALGOUnSupport;
			}
		}
		
		trafficCounterReq.image = image;

		smartmore::ResultCode inferRs = mTrafficCountingEngine.run(trafficCounterReq, trafficCounterRsp);
		if(inferRs != smartmore::ResultCode::Success)
		{
			AlgoMsgError(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage mTrafficCountingEngine.run failed!");
			return ErrALGORunFailed;
		}

		std::vector<std::vector<std::shared_ptr<byte_track::STrack>>> trackResults(mClassesNames.size());

		for (size_t i = 0; i < mClassesNames.size(); ++i)
		{
			if (trafficCounterRsp.detectedBoxes[i].size())
			{
				std::vector<byte_track::Object> detectedObjects;
				for (auto &detectedBox : trafficCounterRsp.detectedBoxes[i])
				{
					byte_track::Object tmpObject(
						{detectedBox.bBox.xMin,
						 detectedBox.bBox.yMin,
						 detectedBox.bBox.xMax - detectedBox.bBox.xMin,
						 detectedBox.bBox.yMax - detectedBox.bBox.yMin},
						detectedBox.classID,
						detectedBox.conf);
					detectedObjects.push_back(tmpObject);
				}
				std::vector<std::shared_ptr<byte_track::STrack>> trackBoxes =
					mTrackers[i].update(detectedObjects);
				trackResults[i] = trackBoxes;
			}
		}


		ALGOVAResult vaResult;
		vaResult.imageInfo = imageInfo;
		vaResult.code = ErrALGOSuccess;
		vaResult.objParams.clear();

		for (size_t i = 0; i < mClassesNames.size(); i++)
		{
			ALGOObjType type = ALGOObjTypeMotor;//TODO		
			list<ALGOObjectParam> objectList;

			if(!trackResults[i].empty())
			{
				for (auto trackResult:trackResults[i])
				{
					ALGOObjectParam object;
					object.boundingBox.x = trackResult->getRect().x();
					object.boundingBox.y = trackResult->getRect().y();
					object.boundingBox.width = trackResult->getRect().width();
					object.boundingBox.height = trackResult->getRect().height();

					object.objectId = trackResult->getTrackId();
					object.confidence = trackResult->getScore();
					object.objLabel = mClassesNames[i];
					object.objType = i;
					//object.propertyList = ;
					//object.roiId = ;
					objectList.emplace_back(object);
				}
			}

			if(!objectList.empty())
			{
				vaResult.objParams[type] = objectList;
			}
		}

		vaResults.emplace_back(vaResult);
	}

	return rs;
}

ErrAlgorithm VideoAnalysis::initEngine()
{
	if(inited)
	{
		return ErrALGOSuccess;
	}

	std::string modelPath("/vms/code/sunth/SmartStream/Algorithm/build/AlgorithmTest/plugins/TrafficCount/model/TrafficCounting.smartmore");
 	smartmore::ResultCode rs = mTrafficCountingEngine.init(modelPath, mGPUId);
	if(rs != smartmore::ResultCode::Success)
	{
		return ErrALGOInitFailed;
	}
	
  	mTrafficCountingEngine.getClassesName(mClassesNames);
	if(mClassesNames.empty())
	{
		return ErrALGOInitFailed;
	}

   	mTrackers = std::vector<byte_track::BYTETracker>(mClassesNames.size(), (10, 10));

	for(auto& className:mClassesNames)
	{
		AlgoMsgDebug(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage start %s!", className.c_str());
	}
	
	inited = true;

	return ErrALGOSuccess;
}