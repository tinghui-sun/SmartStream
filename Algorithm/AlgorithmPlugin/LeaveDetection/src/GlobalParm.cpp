#include "GlobalParm.h"
#include "Poco/Util/IniFileConfiguration.h"
#include "Poco/Exception.h"
#include "Poco/File.h"

using Poco::Util::IniFileConfiguration;
using Poco::AutoPtr;
using Poco::Exception;

GlobalParm& GlobalParm::instance()
{
	static GlobalParm globalParam;
	return globalParam;
}

bool GlobalParm::loadConfig(const PluginParam& param)
{
	m_pluginPath = param.pluginPath;	//插件所在目录绝对路径。目录结构由算法自己决定，建议pluginPath/conf 存放插件配置文件, pluginPath/model 存放模型。	
	m_logger = param.logger;

	string configFilePath;
	AutoPtr<IniFileConfiguration> configRead = new IniFileConfiguration();
	try
	{
		configFilePath = m_pluginPath + "/config/config.ini";
		configRead->load(configFilePath);
	}
	catch (const Exception& e)
	{	
		AlgoMsgError(m_logger, "GlobalParm::loadConfig config file [%s] not exit!", configFilePath.c_str());
		return false;
	}

	m_version = configRead->getString("version", "");
	string modelName = configRead->getString("model", "");

	if(m_version.empty() || modelName.empty())
	{
		AlgoMsgError(m_logger, "GlobalParm::loadConfig config file[%s] invalid", configFilePath.c_str());
		return false;
	}

	m_modlePath = m_pluginPath + "/model/" + modelName;

	Poco::File modleFile(m_modlePath);

	if(!modleFile.exists())
	{
		AlgoMsgError(m_logger, "GlobalParm::loadConfig config file[%s] invalid", m_modlePath.c_str());
		return false;
	}

	return true;
}