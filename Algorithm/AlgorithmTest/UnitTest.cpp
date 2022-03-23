// UnitTest.cpp : 定义控制台应用程序的入口点。
//

#include "CppUnit/TestRunner.h"
#include "SDKUnitTest.h"
#ifdef _DEBUG
#pragma comment(lib, "CppUnitd.lib")
#else
#pragma comment(lib, "CppUnit.lib")
#endif

CppUnitMain(SDKUnitTest)
