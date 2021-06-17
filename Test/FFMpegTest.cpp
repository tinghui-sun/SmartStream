#include "FFMpegTest.h"
#include "CppUnit/TestCaller.h"
#include "CppUnit/TestSuite.h"

extern "C" 
{
	#include <libavcodec/avcodec.h>
	#include <libavformat/avformat.h>
	//#include <libavfilter/avfilter.h">
	#include <libavfilter/buffersink.h>  
	#include <libavfilter/buffersrc.h>
	#include <libavutil/opt.h>
	#include <libavutil/pixdesc.h>
	#include <libavutil/pixfmt.h>
	#include <libavutil/timecode.h>
	#include <libavutil/bprint.h>
	#include <libavutil/time.h>
	#include <libswscale/swscale.h>
}

#include <stdio.h>
#include <string.h>
#include <chrono>
#include <thread> 

#include "ZLToolKit/Util/File.h"
#include "ZLToolKit/Util/util.h"
#include "FFmpegUtil/FFMpegImageLoader.h"
#include "FFmpegUtil/FFMpegImageSaver.h"
#include <iostream>

using namespace toolkit;
using namespace std;

FFMpegTest::FFMpegTest(const std::string& name):
	TestCase(name)
{

}

FFMpegTest::~FFMpegTest()
{

}

void FFMpegTest::loadAndSaveImagesTest()
{
	av_register_all();
	av_log_set_level(AV_LOG_DEBUG);
	string execPath = exeDir();
	string imageFileName = execPath + "201.00_00_02_14.Still003.jpg";
	string outputFile = execPath + "SmartStream_out.jpeg";
	int64_t start_time = av_gettime();
	//AVFrame* frame = OpenImage(imageFileName.c_str());
	//std::cout << av_gettime() - start_time << std::endl;
	//if (frame)
	//{
	//	OperationFrame_EncodeAndWrite_Inner_SaveJpg(frame, outputFile.c_str());
	//}

	int i = 1000;
	do
	{
		FFMpegImageLoader loader;
		FFMpegImageSaver saver;
		AVFrame*  frame = loader.loadImages(imageFileName);
		if (frame)
		{
			std::cout << frame->width << " " << frame->height << std::endl;
			saver.saveImages(outputFile, frame);
		}
		else
		{
			std::cout << "loadImages failed! " << std::endl;
		}
		this_thread::sleep_for(chrono::milliseconds(50));
	} while (false/*--i > 0*/);
}

void FFMpegTest::setUp()
{

}

void FFMpegTest::tearDown()
{

}

CppUnit::Test* FFMpegTest::suite()
{
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite("FFMpegTest");

	CppUnit_addTest(pSuite, FFMpegTest, loadAndSaveImagesTest);

	return pSuite;
}
