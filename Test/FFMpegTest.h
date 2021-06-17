#ifndef COMMON_TEST_H
#define COMMON_TEST_H

#include "CppUnit/TestCase.h"
#include <memory>

using namespace std;

class FFMpegTest :public CppUnit::TestCase,
	public enable_shared_from_this<FFMpegTest>
{
public:
	FFMpegTest(const std::string& name);
	~FFMpegTest();

public:
	void loadAndSaveImagesTest();

public:
	virtual void setUp() override;
	virtual void tearDown() override;

	static CppUnit::Test* suite();
};

#endif