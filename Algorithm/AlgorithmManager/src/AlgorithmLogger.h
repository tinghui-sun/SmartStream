#pragma once
#include "Poco/PatternFormatter.h"
#include "Poco/FormattingChannel.h"
#include "Poco/AsyncChannel.h"
#include "Poco/FileChannel.h"
#include "Poco/Message.h"
#include "Poco/ConsoleChannel.h"
#include "Poco/SplitterChannel.h"
#include "Poco/UnicodeConverter.h"
#include "Poco/AutoPtr.h"
#include "Poco/Message.h"
#include "Poco/Path.h"
#include "Poco/File.h"
#include "Poco/Thread.h"
#include "Poco/Timestamp.h"
#include "Poco/StringTokenizer.h"
#include "Poco/Util/IniFileConfiguration.h"
#include "Poco/NumberParser.h"
#include "Poco/String.h"
#include "Poco/Logger.h"
#include <vector>
#include <string>
#include <memory>
#include <Poco/Path.h>
#include "AlgorithmLog.h"


enum AlgorithmLogType
{
	AlgorithmLogTypeConsole = 1,   //日志输出到控制台
	AlgorithmLogTypeFile = 1 << 1,    //日志输出到文件
	AlgorithmLogTypeAll = AlgorithmLogTypeConsole | AlgorithmLogTypeFile,//同时输出控制台和文件
};

using namespace std;
using namespace Poco;
using namespace Poco::Util;
using namespace Poco::Util;

typedef Poco::AutoPtr<IniFileConfiguration> INIConfigPtr;
typedef Poco::AutoPtr<Logger> LoggerPtr;

class AlgorithmLogger : public Runnable, public AlgoLoggerInterface
{

public:
	AlgorithmLogger(string configFilePath, string logFileDir);

	~AlgorithmLogger();


	/**
	* 初始化日志
	* @moduleName : 程序名
	* @configFilePath : 配置绝对路径
	* @logFileDir : 日志文件写入目录
	* @return 0 初始化成功， 其他失败
	*/
	static shared_ptr<AlgoLoggerInterface>  algorithmLogerInit(const string& configFilePath, const string& logFileDir);

	static shared_ptr<AlgoLoggerInterface> instance();

	virtual void log(AlgoLogPriority level, const string& msg)override;

	virtual void log(AlgoLogPriority level, const char *fmt, ...)override;

	virtual void  forceFlushLog()override;

private:
	

private:
	virtual void run();

	bool init();

	void createLogChannel(std::string& logPath);

	//初始化各个日志模块
	//格式：模块名称:日志级别, 模块名 : 日志级别,
	void createLogModule(std::string& logModules);

	//设置日志格式和日志级别
	void format(std::string module, Message::Priority msgLevel = Message::PRIO_TRACE);

	//如果有默认日志级别，返回默认日志级别。否则返回0
	int analyseLogModules(std::string& logModules, std::map<string, int>& modulesMap);
	
	//设置POCO各个模块日志
	void setLogLevel(LoggerPtr p_pLog, int nLevel);	

	void startWatch();

private:
	AutoPtr<SplitterChannel> m_pSplitter;
	AutoPtr<AsyncChannel> m_pAsyncConsole;
	AutoPtr<AsyncChannel> m_pAsyncFile;
	string m_configFilePath;
	string m_logFileDir;
	File* m_configFile = nullptr;
	AlgorithmLogType m_logType;	//1 输出控制台， 2 输出文件
	Thread m_objWatchThread;
	std::map<string, LoggerPtr> m_objLoggerMap;
};