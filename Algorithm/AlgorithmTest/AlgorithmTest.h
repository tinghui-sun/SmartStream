#ifndef COMMON_TEST_H
#define COMMON_TEST_H

#include "CppUnit/TestCase.h"
#include "AlgorithmManager.h"

class AlgorithmTest :public CppUnit::TestCase,
	public AlgorithmVAListener,
	public enable_shared_from_this<AlgorithmTest>
{
public:
	AlgorithmTest(const std::string& name);
	~AlgorithmTest();

public:
	void LogTest();
	void PluginDemoTest();
	void MotorVehicleStatisticsTest();


private:
	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult) override;

public:
	void setUp();
	void tearDown();

	static CppUnit::Test* suite();
};

#endif