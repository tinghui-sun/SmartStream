#include "SDKTest.h"
#include "SDKUnitTest.h"

CppUnit::Test* SDKUnitTest::suite()
{
	CppUnit::TestSuite* pSuite = new CppUnit::TestSuite("SDKUnitTest");

	pSuite->addTest(SDKTest::suite());

	return pSuite;
}
