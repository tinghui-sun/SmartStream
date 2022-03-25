#include "SDKTest.h"
#include "CppUnit/TestCaller.h"
#include "CppUnit/TestSuite.h"
#include "AlgorithmLog.h"
#include "Poco/Net/NetworkInterface.h"
#include <iostream>
#include <assert.h>
#include <chrono>
#include <thread>
#include <list>
#include <vector>
#include <opencv2/opencv.hpp>

#include <sys/stat.h>
//#include <dirent.h>
#include <iostream> 
#include <sstream>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;
using Poco::Net::NetworkInterface;

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
				imwrite("/home/jiuling/xj/data/tmp/"+string(tmp), img);

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


SDKTest::SDKTest(const std::string& name)
	:CppUnit::TestCase("DeviceSdk")
{
	NetworkInterface::list();
}

SDKTest::~SDKTest()
{

}

void SDKTest::PluginInitTest()
{
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	gpuList.push_back(3);
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypePluginDemo, gpuList);

	assert(mtdPlugin != nullptr);
	
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm();
	auto listener = std::make_shared<VAResListener>();

	mtdAlgo->registerAVResListener(listener);

	assert(mtdAlgo);

	ALGOImageInfo imageInfo;
	std::list<ALGOImageInfo> imagesList;
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};

	mtdPlugin->destoryVAAlgorithm(mtdAlgo);
}

void SDKTest::LogTest()
{
	AlgorithmManager::instances().testLog();
	//Sleep(5000);//等待一步日志写结束
	cout << "LogTest" << endl;
}

void SDKTest::algorithmVAFinished(const std::list <ALGOVAResult>& vaResult)
{

}

void SDKTest::setUp()
{

}

void SDKTest::tearDown()
{

}

void SDKTest::PluginTrYolov5Test()
{
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	gpuList.push_back(3);
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeMotorVehicleStatistics, gpuList);
	assert(mtdPlugin);
	
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm();
	auto listener = std::make_shared<VAResListener>();

	mtdAlgo->registerAVResListener(listener);
	assert(mtdAlgo);

	auto ability =  mtdAlgo->getAbility();
	assert(ability.dataType == ALGOInterfaceParamType_OPENCV_MAT);

	Mat img = imread("/home/jiuling/xj/data/1.png");
	ALGOImageInfo imageInfo;
	imageInfo.imageId = "abcdefghijklmnopqrst0111111";
	imageInfo.imageFormate = ALGOImageFormatCVMat;
	imageInfo.imageWidth = img.cols;
	imageInfo.imageHeigth = img.rows;
	imageInfo.imageBufferType = ALGOBufferCPU;
	imageInfo.imageBufferLen = 0;
	imageInfo.imageBuffer = (char*)img.data;
	std::list<ALGOImageInfo> imagesList;
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};
	imagesList.clear();
	imageInfo.imageId = "abcdefghijklmnopqrst0211111";
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};
	imagesList.clear();
	imageInfo.imageId = "abcdefghijklmnopqrst0311111";
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};
	imagesList.clear();
	imageInfo.imageId = "abcdefghijklmnopqrst0411111";
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};
	imagesList.clear();
	imageInfo.imageId = "abcdefghijklmnopqrst0511111";
	imagesList.push_back(imageInfo);
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	};
	mtdPlugin->destoryVAAlgorithm(mtdAlgo);
	getchar();
	mtdPlugin->pluginRelease();
}

void SDKTest::PluginTrFaceDetectionTest()
{
	remove(faceDetectSaveFile.c_str());
	callbackIndex = 1;
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceDetection, gpuList);
	assert(mtdPlugin);
	
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm();
	assert(mtdAlgo);
	auto listener = std::make_shared<VAResListener>();
	mtdAlgo->registerAVResListener(listener);
	auto ability =  mtdAlgo->getAbility();
	assert(ability.dataType == ALGOInterfaceParamType_OPENCV_MAT);
	//--------------------------
	vector<string> files;
	listFiles("/home/jiuling/xj/data/image", files);
	sort(files.begin(), files.end(), [](string& a, string& b){return a <b; });
	for(int i=0; i<(int)files.size(); i++)
	{
		cout << files[i] << endl;
		Mat img = imread(files[i]);
		ALGOImageInfo imageInfo;
		imageInfo.imageId = files[i];
		imageInfo.imageFormate = ALGOImageFormatCVMat;
		imageInfo.imageWidth = img.cols;
		imageInfo.imageHeigth = img.rows;
		imageInfo.imageBufferType = ALGOBufferCPU;
		imageInfo.imageBufferLen = 0;
		imageInfo.imageBuffer = (char*)img.data;
		std::list<ALGOImageInfo> imagesList;
		imagesList.push_back(imageInfo);
		if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
		{
			cout << "analyzeImage failed!" << endl;
		}
	}
	cout <<"\n\n finished!!!!"<<endl;
	//--------------------------
	getchar();
	mtdPlugin->destoryVAAlgorithm(mtdAlgo);
	getchar();
	mtdPlugin->pluginRelease();
}

struct FaceDet
{
	struct FaecInfo
	{
		float confidence;
		Rect rc;
		vector<float> facials;
	};
	string imgName;
	vector<FaecInfo> infos; 
};
inline void readFaceDetInfo(vector<FaceDet>& faceDets)
{
	ifstream file(faceDetectSaveFile);
	string line = "";
	float fval;
	int ival;
	while(getline(file, line))
	{
		FaceDet det;
		stringstream ss(line);
		ss >> det.imgName;
		while(ss >> fval)
		{
			FaceDet::FaecInfo info;
			info.confidence = fval;
			ss >> info.rc.x >> info.rc.y >> info.rc.width >> info.rc.height;
			for(int i=0;i<10;i++)
			{
				ss >> fval;
				info.facials.push_back(fval);
			}
			det.infos.push_back(info);
		}
		faceDets.push_back(det);
	}
	if(false)
	{
		for(int i=0; i<(int)faceDets.size();i++)
		{
			cout <<i+1<<" "<<faceDets[i].imgName;
			for(int j =0;j<(int)faceDets[i].infos.size(); j++)
			{
				auto& info = faceDets[i].infos[j];
				cout <<"\n  conf:"<< info.confidence << "\n  rect:" << info.rc.x << " "<<info.rc.y<<" "<<info.rc.width<<" "<<info.rc.height<<"\n  faci:";
				for(int k = 0;k<(int)info.facials.size(); k++)
					cout <<info.facials[k]<<"  ";
			}
			cout << endl;
		}
	}
}
void SDKTest::PluginTrFaceRecognitionTest()
{
	vector<FaceDet> faceDets;
	readFaceDetInfo(faceDets);

	callbackIndex = 2;
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceRecognition, gpuList);
	assert(mtdPlugin);

	shared_ptr<AlgorithmIRInterface>  mtdAlgo = mtdPlugin->createIRAlgorithm();
	//auto listener = std::make_shared<VAResListener>();
	//mtdAlgo->registerAVResListener(listener);
	assert(mtdAlgo);
	auto ability =  mtdAlgo->getAbility();
	assert(ability.dataType == ALGOInterfaceParamType_OPENCV_MAT);

	//--------------------------
	for(int i=0;i<(int)faceDets.size();i++)
	{
		auto& det = faceDets[i];
		Mat img = imread(det.imgName), img_dst;
		vector<float> extend;
		for(int j = 0;j<(int)det.infos.size();j++)
		{
			auto& info = det.infos[j];
			//cout <<"\nconf:"<<info.confidence<<"\nrect:"<<info.rc.x <<" "<<info.rc.y<<" "<<info.rc.width<<" "<<info.rc.height<<endl;
			ALGOImageInfo imageInfo;
			imageInfo.imageId = "abcdefghijklmnopqrst03_"+to_string(i+1)+"_"+to_string(j+1);
			imageInfo.imageFormate = ALGOImageFormatCVMat;
			imageInfo.imageWidth = info.rc.width;
			imageInfo.imageHeigth = info.rc.height;
			imageInfo.imageBufferType = ALGOBufferCPU;
			imageInfo.imageBufferLen = 0;
			imageInfo.imageBuffer = (char*)img(info.rc).clone().data;

			extend.clear();
			for(int k=0; k<(int)info.facials.size(); k++)
				info.facials[k] -= k%2==0 ? info.rc.x : info.rc.y;
			extend.insert(extend.end(), info.facials.begin(), info.facials.end());

			extend.emplace_back(1000.0f);//这是测试用来保存图片的的  /home/jiuling/xj/data/tmp1

			imageInfo.extend = (char*)extend.data();
			imageInfo.extendBufferLen = sizeof(float) * extend.size();

			IRFeatureInfo feature;
			if (mtdAlgo->featureExtra(imageInfo, feature) != ErrALGOSuccess)
			{
				cout << "featureExtra failed!" << endl;
			}
			else
			{
				cout<<feature.imageId<<":";
				for(int k=0;k<16;k++)
					cout<<feature.featureBuf[k]<<" ";
				cout<<endl;
			}
		}
	}
	cout <<"\n\n  finish!!!"<<endl;
	//--------------------------
	mtdPlugin->destoryIRAlgorithm(mtdAlgo);
	getchar();
	mtdPlugin->pluginRelease();
}

void SDKTest::PluginTrFaceRecognitionFeatureCompareTest()
{
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceRecognition, gpuList);
	assert(mtdPlugin);

	shared_ptr<AlgorithmIRInterface>  mtdAlgo = mtdPlugin->createIRAlgorithm();
	assert(mtdAlgo);

	string src = ",1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,1.0,2.0,3.0,";
	string dst = "0.5,1.0,1.5,0.2,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,0.5,1.0,1.5,";
	string dst1 = "0.5,1.0,1.5,0.2,1.0,1.5,0.5,1.0,1.1,0.2,1.6,1.1,0.5,1.4,1.6,0.2,1.3,1.8,";
	float sim = 0.0f;
	mtdAlgo->compare(src,src, sim); cout<<sim<<endl;
	mtdAlgo->compare(src,dst, sim); cout<<sim<<endl;
	mtdAlgo->compare(src,dst1, sim); cout<<sim<<endl;

	cout<<"----------------------+++++++++++++++++++++"<<endl;
	list<IRFeatureInfo> srcFeature, dstFeature;
	float threshold = 0.75f;
	uint32_t limit = 10;
	list<IRCompareResult> result;
	/*{
		IRFeatureInfo info;
		info.imageId = "srcFeature_"+to_string(0);
		info.featureLen = info.featureIndexLen = 10;
		info.featureBuf[0] = 1.0f; info.featureBuf[1] = 1.0f; info.featureBuf[2] = 1.0f; info.featureBuf[3] = 1.0f; info.featureBuf[4] = 1.0f;
		info.featureBuf[5] = 1.0f; info.featureBuf[6] = 1.0f; info.featureBuf[7] = 1.0f; info.featureBuf[8] = 1.0f; info.featureBuf[9] = 1.0f;
		srcFeature.push_back(info);
	}
	{
		IRFeatureInfo info;
		info.imageId = "dstFeature_"+to_string(0);
		info.featureLen = info.featureIndexLen = 11;
		info.featureBuf[0] = 1.0f; info.featureBuf[1] = 1.0f; info.featureBuf[2] = 1.0f; info.featureBuf[3] = 1.0f; info.featureBuf[4] = 1.0f;
		info.featureBuf[5] = 1.0f; info.featureBuf[6] = 1.0f; info.featureBuf[7] = 1.0f; info.featureBuf[8] = 1.0f; info.featureBuf[9] = 1.0f; 
		info.featureBuf[10] = 1.0f;
		dstFeature.push_back(info);

		info.imageId = "dstFeature_"+to_string(1);
		info.featureLen = info.featureIndexLen = 10;
		info.featureBuf[0] = 1.2f; info.featureBuf[1] = 1.5f; info.featureBuf[2] = 1.0f; info.featureBuf[3] = 1.1f; info.featureBuf[4] = 1.4f;
		info.featureBuf[5] = 1.0f; info.featureBuf[6] = 1.6f; info.featureBuf[7] = 1.7f; info.featureBuf[8] = 1.0f; info.featureBuf[9] = 1.0f; 
		dstFeature.push_back(info);
	}*/
	srand(time(0));
	for(int i = 0; i< 1; i++) 
	{
		IRFeatureInfo info;
		info.imageId = "srcFeature_"+to_string(i);
		info.featureLen = info.featureIndexLen = 512;
		for(int j =0;j<info.featureLen;j++)
		{
			info.featureBuf[j] = (rand() % 100) / 100.0f;
			info.featureIndex[j] = (rand() % 100) / 100.0f;
		}
		srcFeature.push_back(info);
	}
	for(int i = 0; i< 1; i++) 
	{
		IRFeatureInfo info;
		info.imageId = "dstFeature_"+to_string(i);
		info.featureLen = info.featureIndexLen = 512;
		for(int j =0;j<info.featureLen;j++)
		{
			info.featureBuf[j] = (rand() % 100) / 100.0f;
			info.featureIndex[j] = (rand() % 100) / 100.0f;
		}
		dstFeature.push_back(info);
	}
	mtdAlgo->compare(IRMatchTypeLong, srcFeature, dstFeature, threshold, limit, result);
	int index=1;
	for(auto& node: result)
        printf("compare result: %05d %s %s %f\n", index++, node.srcImageId.c_str(), node.dstImageId.c_str(), node.similarity);
	mtdPlugin->destoryIRAlgorithm(mtdAlgo);
	getchar();
	mtdPlugin->pluginRelease();
}

void SDKTest::PluginTrFaceDRCTest()
{
	callbackIndex = 3;
	std::list<int> gpuList;
	gpuList.push_back(1);
	gpuList.push_back(2);
	AlgorithmManager::instances().initManager();

	AlgorithmPluginInterface* mtdPluginDet = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceDetection, gpuList);
	assert(mtdPluginDet);
	shared_ptr<AlgorithmVAInterface>  mtdAlgoDet = mtdPluginDet->createVAAlgorithm();
	assert(mtdAlgoDet);
	auto listener = std::make_shared<VAResListener>();
	mtdAlgoDet->registerAVResListener(listener);
	assert(mtdAlgoDet->getAbility().dataType == ALGOInterfaceParamType_OPENCV_MAT);
	
	AlgorithmPluginInterface* mtdPluginRec = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceRecognition, gpuList);
	assert(mtdPluginRec);
	shared_ptr<AlgorithmIRInterface>  mtdAlgoRec = mtdPluginRec->createIRAlgorithm();
	assert(mtdAlgoRec);
	assert(mtdAlgoRec->getAbility().dataType == ALGOInterfaceParamType_OPENCV_MAT);

	//--------------------------
	cout<<"输入非e的字符继续，否则退出！！！"<<endl;
	char c=0;
	while((c = getchar())!='e')
	{
		string file1, file2;
		cout<<"输入第一个文件："<<endl;
		getline(cin, file1);
		cout<<"输入第二个文件："<<endl;
		getline(cin, file2);
		cout<<"file1:"<<file1<<"  file2:"<<file2<<endl;
		vector<string> files;
		files.push_back(file1);
		files.push_back(file2);

		for(auto& file: files)
		{
			DetResults[file] = DetResult();
			Mat img = imread(file);
			ALGOImageInfo imageInfo;
			imageInfo.imageId = file;
			imageInfo.imageFormate = ALGOImageFormatCVMat;
			imageInfo.imageWidth = img.cols;
			imageInfo.imageHeigth = img.rows;
			imageInfo.imageBufferType = ALGOBufferCPU;
			imageInfo.imageBufferLen = 0;
			imageInfo.imageBuffer = (char*)img.data;
			std::list<ALGOImageInfo> imagesList;
			imagesList.push_back(imageInfo);
			if (mtdAlgoDet->analyzeImage(imagesList) != ErrALGOSuccess)
				cout << "analyzeImage failed!" << endl;
		}
		while(true)
		{
			int valid = 0;
			for(auto& file: files)
			{
				if(DetResults[file].valid == true)
					valid+=1;
			}
			if(valid == (int)files.size())
				break;
			else
				this_thread::yield();
		}
		cout<<"detection finish!!"<<endl;
		int index = 0;
		list<IRFeatureInfo> srcFeature, dstFeature;
		float threshold = 0.0f;
		uint32_t limit = 10;
		list<IRCompareResult> result;
		for(auto& file: files)
		{
			auto& ret = DetResults[file];
			Mat img = imread(file);
			/*rectangle(img, ret.rc, Scalar(0, 255, 0), 1);
			for(int i=0;i<(int)ret.pt.size();)
			{
				cv::Point pt(ret.pt[i], ret.pt[i+1]);
				cv::circle(img, pt, 1, cv::Scalar(255, 0, 0), 1);
				i+=2;
			}
			auto _file = "/home/jiuling/vms/cbb/Algorithm/AlgorithmManager/test/"+to_string(index)+".jpg";
			imwrite(_file, img);
			cout<<"save "<<_file<<endl;*/

			ALGOImageInfo imageInfo;
			imageInfo.imageId = file;
			imageInfo.imageFormate = ALGOImageFormatCVMat;
			imageInfo.imageWidth = ret.rc.width;
			imageInfo.imageHeigth = ret.rc.height;
			imageInfo.imageBufferType = ALGOBufferCPU;
			imageInfo.imageBufferLen = 0;
			imageInfo.imageBuffer = (char*)img(ret.rc).clone().data;

			vector<float> extend;
			for(int k=0; k<(int)ret.pt.size(); k++)
				ret.pt[k] -= k%2==0 ? ret.rc.x : ret.rc.y;
			extend.insert(extend.end(), ret.pt.begin(), ret.pt.end());

			//extend.emplace_back(1000.0f);//这是测试用来保存图片的的  /home/jiuling/xj/data/tmp1

			imageInfo.extend = (char*)extend.data();
			imageInfo.extendBufferLen = sizeof(float) * extend.size();

			IRFeatureInfo feature;
			if (mtdAlgoRec->featureExtra(imageInfo, feature) != ErrALGOSuccess)
				cout << "featureExtra failed!" << endl;
			if(index==0)
				srcFeature.push_back(feature);
			else
				dstFeature.push_back(feature);
			index++;
		}
		mtdAlgoRec->compare(IRMatchTypeLong, srcFeature, dstFeature, threshold, limit, result);
		for(auto& node: result)
        	printf("compare result: %s %s %f\n", node.srcImageId.c_str(), node.dstImageId.c_str(), node.similarity);
	}
	mtdPluginDet->destoryVAAlgorithm(mtdAlgoDet);
	mtdPluginRec->destoryIRAlgorithm(mtdAlgoRec);
	mtdPluginDet->pluginRelease();
	mtdPluginRec->pluginRelease();
}

void SDKTest::VideoQualtyTest()
{
	std::list<int> gpuList; 
	gpuList.push_back(1);
	gpuList.push_back(2);
	gpuList.push_back(3);  

	AlgorithmManager::instances().initManager();
	AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeVideoQualityDetection, gpuList);

	assert(mtdPlugin != nullptr);
	
	shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm();
	
	//auto listener = std::make_shared<VAResListener>();
	//mtdAlgo->registerAVResListener(listener);
 
	assert(mtdAlgo); 
     
	auto ability =  mtdAlgo->getAbility();
	assert(ability.dataType == ALGOInterfaceParamType_OPENCV_MAT);
 
	Mat img = imread("/home/jiuling/xj/data/val2017/000000000139.jpg");
	ALGOImageInfo imageInfo;
	imageInfo.imageId = "abcdefghijklmnopqrst0111111";
	imageInfo.imageFormate = ALGOImageFormatCVMat;
	imageInfo.imageWidth = img.cols;
	imageInfo.imageHeigth = img.rows;
	imageInfo.imageBufferType = ALGOBufferCPU;
	imageInfo.imageBufferLen = 0;
	imageInfo.imageBuffer = (char*)img.data;
	std::list<ALGOImageInfo> imagesList;

	imagesList.push_back(imageInfo);
	int i=0;
    for( i =0 ; i< 17; i++)
	{
	    imageInfo.imageId = "abcdefghijklmnopqrst0211111" + to_string(i);
	    imagesList.push_back(imageInfo);
	}
	if (mtdAlgo->analyzeImage(imagesList) != ErrALGOSuccess)
	{
		cout << "analyzeImage failed!" << endl;
	}
	imagesList.clear();
	mtdPlugin->destoryVAAlgorithm(mtdAlgo);
	getchar();
	mtdPlugin->pluginRelease();
}


void SDKTest::PTest()
{
	cout << "----------PTest----------" << endl;

	std::list<int> gpuList;
	gpuList.push_back(1);
	AlgorithmManager::instances().initManager();
	
	auto c = getchar();
	cout<<"\n\n"<<endl;
	if(c!='a')
	{
		AlgorithmPluginInterface* mtdPlugin = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceDetection, gpuList);
		shared_ptr<AlgorithmVAInterface>  mtdAlgo = mtdPlugin->createVAAlgorithm();
		cout<<"++---1 mtdPlugin:"<<mtdPlugin<<endl;
		cout<<"++---1 mtdAlgo:"<<mtdAlgo<<endl;
	}
	cout<<"\n\n"<<endl;
	if(c!='b')
	{
		gpuList.push_back(2);
		AlgorithmPluginInterface* mtdPlugin1 = AlgorithmManager::instances().getAlgorithmPlugin(ALGOTypeFaceRecognition, gpuList);
		shared_ptr<AlgorithmIRInterface>  mtdAlgo1 = mtdPlugin1->createIRAlgorithm();
		cout<<"++---2 mtdPlugin1:"<<mtdPlugin1<<endl;
		cout<<"++---2 mtdAlgo1:"<<mtdAlgo1<<endl;
	}
	return;
}


CppUnit::Test* SDKTest::suite()
{
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite("AlgorithmTest");

	CppUnit_addTest(pSuite, SDKTest, PluginInitTest);
	//CppUnit_addTest(pSuite, SDKTest, LogTest);
	CppUnit_addTest(pSuite, SDKTest, PTest);
	CppUnit_addTest(pSuite, SDKTest, PluginTrYolov5Test);

	CppUnit_addTest(pSuite, SDKTest, PluginTrFaceDetectionTest);
	CppUnit_addTest(pSuite, SDKTest, PluginTrFaceRecognitionTest);

	CppUnit_addTest(pSuite, SDKTest, PluginTrFaceRecognitionFeatureCompareTest);
	CppUnit_addTest(pSuite, SDKTest, PluginTrFaceDRCTest);

	CppUnit_addTest(pSuite, SDKTest, VideoQualtyTest);
	return pSuite;
}

