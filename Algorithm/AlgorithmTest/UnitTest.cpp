// UnitTest.cpp : �������̨Ӧ�ó������ڵ㡣
//
//export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/../../3rdpart/Linux/Poco/lib/:$(pwd)/../../3rdpart/Linux/opencv/lib/

#include "CppUnit/TestRunner.h"
#include "AlgorithmUnitTest.h"
#ifdef _DEBUG
#pragma comment(lib, "CppUnitd.lib")
#else
#pragma comment(lib, "CppUnit.lib")
#endif

CppUnitMain(AlgorithmUnitTest)
