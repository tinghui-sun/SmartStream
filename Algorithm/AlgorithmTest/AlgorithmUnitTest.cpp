#include "AlgorithmTest.h"
#include "AlgorithmUnitTest.h"

CppUnit::Test* AlgorithmUnitTest::suite()
{
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite("AlgorithmUnitTest");

	pSuite->addTest(AlgorithmTest::suite());

	return pSuite;
}
