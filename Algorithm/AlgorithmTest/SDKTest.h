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
	void LogTest();
	void PTest();
	void MotorVehicleStatistics();

private:
	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult) override;

public:
	void setUp();
	void tearDown();

	static CppUnit::Test* suite();
};

#endif