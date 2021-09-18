#include "UnitTest.h"
#include "FFMpegTest.h"
#include "TensorrtTest.h"
#include "CppUnit/TestRunner.h"

#if defined(_DEBUG)
#pragma comment(lib, "flvd.lib")
#pragma comment(lib, "movd.lib")
#pragma comment(lib, "mpegd.lib")
#pragma comment(lib, "zlmediakitd.lib")
#pragma comment(lib, "zltoolkitd.lib")
#pragma comment(lib, "mk_apid.lib")
#pragma comment(lib, "CppUnitd.lib")
#else
#pragma comment(lib, "flv.lib")
#pragma comment(lib, "mov.lib")
#pragma comment(lib, "mpeg.lib")
#pragma comment(lib, "zlmediakit.lib")
#pragma comment(lib, "zltoolkit.lib")
#pragma comment(lib, "mk_api.lib")
#pragma comment(lib, "CppUnitmd.lib")
#endif

CppUnit::Test* SmartStreamUnitTest::suite()
{

	string testName("FFMpegTest");
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite();

	pSuite->addTest(FFMpegTest::suite());
	pSuite->addTest(TensorrtTest::suite());

	return pSuite;
};

CppUnitMain(SmartStreamUnitTest)
