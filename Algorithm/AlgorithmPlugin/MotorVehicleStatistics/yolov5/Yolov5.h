#pragma once
#include "BaseModel.h"

class Yolo5 : public BaseModel
{
public:
	Yolo5();
	virtual ~Yolo5();

public:
	virtual bool initialize(const char* configPath);

protected:
	string getClassName() { return "Yolo5"; }

private:
	//ģ�͹������   ��Ҫ��д����
	bool createNetwork(INetworkDefinition* network, map<string, Weights>& weightMap);
	void createNetworkNormal(INetworkDefinition* network, map<string, Weights>& weightMap, DataType dt, float& gd, float& gw);
	void createNetworkNormalP6(INetworkDefinition* network, map<string, Weights>& weightMap, DataType dt, float& gd, float& gw);

};

