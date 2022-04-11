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
	string m_pluginPath;	//插件所在目录绝对路径。目录结构由算法自己决定，建议pluginPath/conf 存放插件配置文件, pluginPath/model 存放模型。
	list<int> m_gpuList;	//算法使用哪些GPU，为空使用CPU
	shared_ptr<AlgoLoggerInterface> m_logger;
	string m_version;
	string m_modlePath;
};