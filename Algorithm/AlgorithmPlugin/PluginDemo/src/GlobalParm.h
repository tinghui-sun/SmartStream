#pragma once
#include <string>
#include "Algorithm.h"
#include "AlgorithmLog.h"

using namespace std;
class GlobalParm
{
public:	
	virtual ~GlobalParm() {};
	static GlobalParm& instance();
	bool loadConfig(const PluginParam& param);

private:
	GlobalParm() {};	

public:
	string m_pluginPath;	//�������Ŀ¼����·����Ŀ¼�ṹ���㷨�Լ�����������pluginPath/conf ��Ų�������ļ�, pluginPath/model ���ģ�͡�
	shared_ptr<AlgoLoggerInterface> m_logger;
	string m_version;
	string m_modlePath;
};