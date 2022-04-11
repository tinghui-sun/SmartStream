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
		smartmore::DetectionRequest req;
        smartmore::DetectionResponse rsp;

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
		
		req.threshold = 0.4f;
        req.image = image;

		smartmore::ResultCode inferRs = mEngine.Run(req, rsp);
		if(inferRs != smartmore::ResultCode::Success)
		{
			AlgoMsgError(GlobalParm::instance().m_logger, "VideoAnalysis::analyzeImage mTrafficCountingEngine.run failed!");
			return ErrALGORunFailed;
		}


		ALGOVAResult vaResult;
		vaResult.imageInfo = imageInfo;
		vaResult.code = ErrALGOSuccess;
		vaResult.objParams.clear();
		
		list<ALGOObjectParam> objectList;

        for (size_t i = 0; i < rsp.box_list.size(); ++i)
        {
            std::cout << "[" << i << "] id[" << rsp.box_list[i].label_id << "] name["
                    << rsp.box_list[i].label_name << "] "
                    << rsp.box_list[i].score << ":"
                    << "(" << rsp.box_list[i].xmin << ","
                    << rsp.box_list[i].xmax << ","
                    << rsp.box_list[i].ymin << ","
                    << rsp.box_list[i].ymax << ")" << std::endl;

			ALGOObjectParam object;
			object.objectId = 0;
			object.confidence = rsp.box_list[i].score;
			object.objLabel = rsp.box_list[i].label_name;
			object.objType = rsp.box_list[i].label_id;
			object.boundingBox.x = rsp.box_list[i].xmin;
			object.boundingBox.y = rsp.box_list[i].ymin;
			object.boundingBox.width = rsp.box_list[i].xmax - rsp.box_list[i].xmin;
			object.boundingBox.height = rsp.box_list[i].ymax - rsp.box_list[i].ymin;

			if (rsp.box_list[i].label_id == 1)
			{//Àë¸Ú

			}
			else
			{//ÔÚ¸Ú

			}

			objectList.emplace_back(object);
        }

		if(!objectList.empty())
		{
			vaResult.objParams[ALGOObjTypeBody] = objectList;
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

	std::string modelPath(GlobalParm::instance().m_modlePath);
 	smartmore::ResultCode rs = mEngine.Init(modelPath, false, 0);
	if(rs != smartmore::ResultCode::Success)
	{
		return ErrALGOInitFailed;
	}
	
	inited = true;
	return ErrALGOSuccess;
}