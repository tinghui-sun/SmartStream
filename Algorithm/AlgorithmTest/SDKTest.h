#ifndef COMMON_TEST_H
#define COMMON_TEST_H

#include "CppUnit/TestCase.h"
#include "AlgorithmManager.h"

class SDKTest :public CppUnit::TestCase,
	public AlgorithmVAListener,
	public enable_shared_from_this<SDKTest>
{
public:
	SDKTest(const std::string& name);
	~SDKTest();

public:
	void PluginInitTest();

	void PluginTrYolov5Test();

	void PluginTrFaceDetectionTest();
	void PluginTrFaceRecognitionTest();
	void PluginTrFaceRecognitionFeatureCompareTest(); 
	void PluginTrFaceDRCTest();

	void VideoQualtyTest();
	void LogTest();
	void PTest();

private:
	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult) override;

public:
	void setUp();
	void tearDown();

	static CppUnit::Test* suite();
};

#endif