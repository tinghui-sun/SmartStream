#include "AlgorithmTest.h"
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <list>
#include <vector>

#include "CppUnit/TestCaller.h"
#include "CppUnit/TestSuite.h"
#include "opencv2/opencv.hpp"
#include "cuda_runtime_api.h"
#include "cuda.h"

#include <sys/stat.h>
//#include <dirent.h>
#include <iostream> 
#include <sstream>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

static int callbackIndex = 0;
static string faceDetectSaveFile = "/home/jiuling/xj/data/facedetect.txt";

struct DetResult
{
	bool valid{false};
	Rect rc;
	vector<float> pt;
};
static map<string, DetResult> DetResults;

inline void listFiles(const string path, vector<string>& files)
{
	//DIR *dir = nullptr;
	//struct dirent *ptr = nullptr;
	//if ((dir = opendir(path.c_str())) == nullptr)
	//{
	//	return;
	//}
	//while ((ptr = readdir(dir)) != nullptr)
	//{
	//	if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
	//		continue;
	//	else if (ptr->d_type == 8)    //file  
	//		files.push_back(path + "/" + string(ptr->d_name));
	//	else if (ptr->d_type == 10)    //link file  
	//		continue;
	//	else if (ptr->d_type == 4)    //dir  
	//	{
	//		string _path = path + "/" + string(ptr->d_name);
	//		listFiles(_path, files);
	//	}
	//}
	//closedir(dir);
}

template <class Type> 
Type stringToNum(const string str) 
{ 
	istringstream iss(str); 
	Type num; iss >> num; 
	return num; 
}



class VAResListener : public AlgorithmVAListener
{
public:
	VAResListener() {};
	virtual ~VAResListener() {};

	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult)override
	{
		cout << "algorithmVAFinished " << vaResult.size() << endl;

		if(callbackIndex == 1)
		{
			static int index = 0;
			for(auto& info: vaResult)
			{
				string buf = ""; 
				buf +=info.imageInfo.imageId;
				Mat img = imread(info.imageInfo.imageId);
				for(auto& fd: info.objParams)
				{
					/*cv::putText(img, to_string((int)(fd.score*100)), cv::Point(fd.boundingBox.x + 5, fd.boundingBox.y + 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
					rectangle(img, cv::Rect(fd.boundingBox.x, fd.boundingBox.y, fd.boundingBox.width, fd.boundingBox.height), Scalar(0, 255, 0), 1);
					buf+= " "+to_string(fd.score)+" "+to_string(fd.boundingBox.x)+" "+to_string(fd.boundingBox.y)+" "+to_string(fd.boundingBox.width)+" "+to_string(fd.boundingBox.height);

					int _index = 0;
					float tmp;
					for(auto& pro: fd.propertyList)
					{
						_index++;
						if(_index % 2 ==0)
						{
							cv::Point pt(tmp, stringToNum<float>(pro.propertyValue));
							cv::circle(img, pt, 1, cv::Scalar(255, 0, 0), 1);
							buf +=" "+to_string(pt.x)+" "+to_string(pt.y);
						}
						else
							tmp = stringToNum<float>(pro.propertyValue);
					}*/
				}
				buf+="\n";

				char tmp[20]={0};
				sprintf(tmp, "%04d.jpg", index++);
				imwrite("/home/tmp/"+string(tmp), img);

				ofstream file(faceDetectSaveFile, ios::binary|ios::app);
				if (file.is_open())
				{
					file.write(buf.c_str(), buf.length());
					file.close();
				}
			}
		}
		else if(callbackIndex == 2)
		{

		}
		else if(callbackIndex == 3)
		{
			//for(auto& info: vaResult)
			//{
			//	auto det = &DetResults[info.imageId];
			//	det->valid = true;
			//	for(auto& fd: info.objParamList)
			//	{
			//		det->rc = Rect(fd.boundingBox.x, fd.boundingBox.y, fd.boundingBox.width, fd.boundingBox.height);
			//		for(auto& pro: fd.propertyList)
			//			det->pt.push_back(stringToNum<float>(pro.propertyValue));
			//	}
			//	cout<<info.imageId<<" return!!"<<endl;
			//	break;
			//}
		}
		else
		{
			for (auto vaImage: vaResult)
			{
				cout << "algorithmVAFinished " << vaImage.imageInfo.imageId << endl;
			}	
		}	
	}
};


AlgorithmTest::AlgorithmTest(const std::string& name)
	:CppUnit::TestCase("AlgorithmTest")
{

}

AlgorithmTest::~AlgorithmTest()
{

}

void AlgorithmTest::PluginDemoTest()
{
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypePluginDemo, 0);

	assert(mtdPlugin != nullptr);
	
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm(0);
	auto listener = std::make_shared<VAResListener>();

	mtdAlgo->registerAVResListener(listener);

	assert(mtdAlgo);

	ALGOImageInfo imageInfo;
	std::list<ALGOImageInfo> imagesList;
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImageASync(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	if (mtdAlgo->analyzeImageASync(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	if (mtdAlgo->analyzeImageASync(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	mtdPlugin->destoryVAAlgorithm(mtdAlgo);
}

void AlgorithmTest::LogTest()
{
	AlgorithmManager::instances().testLog();
	cout << "LogTest" << endl;
}

void AlgorithmTest::algorithmVAFinished(const std::list <ALGOVAResult>& vaResult)
{

}

#define GPUTest
void AlgorithmTest::MotorVehicleStatisticsTest()
{
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeMotorVehicleStatistics, 0);

	assert(mtdPlugin != nullptr);

	int gpuId = 0;
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm(gpuId);
	assert(mtdAlgo);

	auto listener = std::make_shared<VAResListener>();
	mtdAlgo->registerAVResListener(listener);


	std::list<ALGOImageInfo> imageList;
 	std::list <ALGOVAResult> vaResult;

	cv::Mat image = cv::imread("/Algorithm/AlgorithmPlugin/TrafficCount/TrafficCount/images/image_00000001.jpg");

	int rows = image.rows;
	int cols = image.cols;
	int num_el = rows*cols;
	int totalSize1 = num_el*image.elemSize();


	int totalSize2 = image.total()*image.elemSize();

	cout << "len1 " << totalSize1 << " len2 " << totalSize2;

	ALGOImageInfo info;
	info.imageBufferLen = totalSize2;	
	info.imageFormate = ALGOImageFormatCVMat;
	info.imageWidth =  image.cols;
	info.imageHeight = image.rows;
	info.imageId = "1";

#ifdef GPUTest
	void* cudaBuff = nullptr;
	cudaSetDevice(gpuId);	
	cudaMalloc(&cudaBuff, totalSize1);	
	auto cudaErr = cudaMemcpy(cudaBuff, image.data, totalSize1, cudaMemcpyHostToDevice);	
	info.imageBuffer = (char*)cudaBuff;
	info.imageBufferType = ALGOBufferGPU;
#else
	info.imageBuffer = (char*)image.data;
	info.imageBufferType = ALGOBufferCPU;
#endif

	imageList.emplace_back(info);
	ErrAlgorithm rs = mtdAlgo->analyzeImageSync(imageList, vaResult);


	if(rs == ErrALGOSuccess)
	{
		for(auto vaRs : vaResult)
		{		
			stringstream rsStr;	
			for(auto objParamsIt : vaRs.objParams)
			{
				rsStr << vaRs.code << " ";
				rsStr << objParamsIt.first << " ";
				rsStr << "\n";

				for(auto objInfo:objParamsIt.second)
				{
					rsStr << objInfo.objectId << " ";
					rsStr << objInfo.objType << " ";
					rsStr << objInfo.objLabel << " ";
					rsStr << objInfo.boundingBox.width << " ";
					rsStr << objInfo.boundingBox.height << " ";

					for(auto property : objInfo.propertyList)
					{
						rsStr << property.propertyName << " ";
						rsStr << property.propertyValue << " ";
					}

					rsStr << "\n";
				}
			}

			cout << rsStr.str() << endl;
		}
	}

	mtdPlugin->destoryVAAlgorithm(mtdAlgo);

#ifdef GPUTest
	cudaFree(cudaBuff);
#endif
}


void AlgorithmTest::LeaveDetectionTest()
{
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeLeaveDetection, 0);

	assert(mtdPlugin != nullptr);

	int gpuId = 0;
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm(gpuId);
	assert(mtdAlgo);

	auto listener = std::make_shared<VAResListener>();
	mtdAlgo->registerAVResListener(listener);


	std::list<ALGOImageInfo> imageList;
 	std::list <ALGOVAResult> vaResult;

	cv::Mat image = cv::imread("/Algorithm/AlgorithmPlugin/TrafficCount/TrafficCount/images/image_00000001.jpg");

	int rows = image.rows;
	int cols = image.cols;
	int num_el = rows*cols;
	int totalSize1 = num_el*image.elemSize();


	int totalSize2 = image.total()*image.elemSize();

	cout << "len1 " << totalSize1 << " len2 " << totalSize2;

	ALGOImageInfo info;
	info.imageBufferLen = totalSize2;	
	info.imageFormate = ALGOImageFormatCVMat;
	info.imageWidth =  image.cols;
	info.imageHeight = image.rows;
	info.imageId = "1";

#ifdef GPUTest
	void* cudaBuff = nullptr;
	cudaSetDevice(gpuId);	
	cudaMalloc(&cudaBuff, totalSize1);	
	auto cudaErr = cudaMemcpy(cudaBuff, image.data, totalSize1, cudaMemcpyHostToDevice);	
	info.imageBuffer = (char*)cudaBuff;
	info.imageBufferType = ALGOBufferGPU;
#else
	info.imageBuffer = (char*)image.data;
	info.imageBufferType = ALGOBufferCPU;
#endif

	imageList.emplace_back(info);
	ErrAlgorithm rs = mtdAlgo->analyzeImageSync(imageList, vaResult);


	if(rs == ErrALGOSuccess)
	{
		for(auto vaRs : vaResult)
		{		
			stringstream rsStr;	
			for(auto objParamsIt : vaRs.objParams)
			{
				rsStr << vaRs.code << " ";
				rsStr << objParamsIt.first << " ";
				rsStr << "\n";

				for(auto objInfo:objParamsIt.second)
				{
					rsStr << objInfo.objectId << " ";
					rsStr << objInfo.objType << " ";
					rsStr << objInfo.objLabel << " ";
					rsStr << objInfo.boundingBox.width << " ";
					rsStr << objInfo.boundingBox.height << " ";

					for(auto property : objInfo.propertyList)
					{
						rsStr << property.propertyName << " ";
						rsStr << property.propertyValue << " ";
					}

					rsStr << "\n";
				}
			}

			cout << rsStr.str() << endl;
		}
	}

	mtdPlugin->destoryVAAlgorithm(mtdAlgo);

#ifdef GPUTest
	cudaFree(cudaBuff);
#endif
}

void AlgorithmTest::setUp()
{

}

void AlgorithmTest::tearDown()
{

}

CppUnit::Test* AlgorithmTest::suite()
{
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite("AlgorithmTest");

	CppUnit_addTest(pSuite, AlgorithmTest, PluginDemoTest);
	CppUnit_addTest(pSuite, AlgorithmTest, MotorVehicleStatisticsTest);
	CppUnit_addTest(pSuite, AlgorithmTest, LeaveDetectionTest);
	CppUnit_addTest(pSuite, AlgorithmTest, LogTest);
	return pSuite;
}

