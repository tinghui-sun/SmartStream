#include "GlobalParm.h"
#include "Poco/Util/IniFileConfiguration.h"
#include "Poco/Exception.h"

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

	//string configFilePath;
	//AutoPtr<IniFileConfiguration> configRead = new IniFileConfiguration();
	//try
	//{
	//	configFilePath = m_pluginPath + "/conf/config.ini";
	//	configRead->load(configFilePath);
	//}
	//catch (const Exception& e)
	//{	
	//	AlgoMsgError(m_logger, "GlobalParm::loadConfig config file [%s] not exit!", configFilePath.c_str());
	//	return false;
	//}

	//m_serverIp = configRead->getString("ServerIp", "127.0.0.1");
	//m_serverPort = configRead->getInt("ServerPort", 9099);
	return true;
}