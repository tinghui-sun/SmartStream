#include "AlgorithmLogger.h"

using Poco::Message;
using Poco::PatternFormatter;
using Poco::FormattingChannel;
using Poco::FileChannel;
using Poco::AsyncChannel;
using Poco::ConsoleChannel;
using Poco::SplitterChannel;
using Poco::AutoPtr;
using Poco::UnicodeConverter;
using Poco::Path;
using Poco::File;
using Poco::Thread;
using Poco::Runnable;
using Poco::Timestamp;
using Poco::Util::IniFileConfiguration;
using Poco::StringTokenizer;
using Poco::NumberParser;
using Poco::Logger;

using namespace std;

const static string logModuleName = "algorithm";
shared_ptr<AlgorithmLogger> gLogger = nullptr;
AlgorithmLogger::AlgorithmLogger(string configFilePath,string logFileDir)
	: m_configFilePath(configFilePath),
	  m_logFileDir(logFileDir)
{
	if (!init())
	{
		std::cout << "AlgorithmLog init failed!" << std::endl;
		abort();
	}
}


AlgorithmLogger::~AlgorithmLogger()
{

}

shared_ptr<AlgoLoggerInterface> AlgorithmLogger::instance()
{
	if (gLogger)
	{
		return gLogger;
	}
	else
	{
		cout << "call algorithmLogerInit first!" << endl;
		return nullptr;
	}	
}

void AlgorithmLogger::forceFlushLog()
{
#if 1
	//开启flush
	FormattingChannel* pFChannel = dynamic_cast<FormattingChannel*>(m_pAsyncFile->getChannel());
	pFChannel->getChannel()->setProperty("flush", "true");

	//刷新所有日志
	std::map<string, LoggerPtr>::iterator iterLog = m_objLoggerMap.begin();
	for (; iterLog != m_objLoggerMap.end(); ++iterLog)
	{
		iterLog->second->trace("AlgorithmLogger::forceFlushLog now!");
	}

	//关闭flush
	pFChannel->getChannel()->setProperty("flush", "false");
#endif
}

void AlgorithmLogger::startWatch()
{
	m_objWatchThread.start(*this);
}

void AlgorithmLogger::format(std::string module, Message::Priority msgLevel)
{
	LoggerPtr curLogger = &Logger::get(module);
	//default level
	curLogger->setLevel(msgLevel);
	curLogger->setChannel(m_pSplitter);

	m_objLoggerMap[module] = curLogger;
}

void AlgorithmLogger::createLogChannel(std::string& logPath)
{
	AutoPtr<PatternFormatter> formatter = new PatternFormatter("%Y-%m-%d %H:%M:%S%.%i %t");
	AutoPtr<FormattingChannel> consoleFormatChannel = new FormattingChannel(formatter);
	consoleFormatChannel->setChannel(new ConsoleChannel);

	//file log setting
	AutoPtr<FormattingChannel> fileformatChannel = new FormattingChannel(formatter);
	formatter->setProperty(PatternFormatter::PROP_TIMES, "local");

	//init File Property for log
	AutoPtr<FileChannel> fileChannel = new FileChannel(logPath);

	//默认不每条日志刷新磁盘
	fileChannel->setProperty("flush", "false");

	//reserver file size of each log file
	fileChannel->setProperty(FileChannel::PROP_ROTATION, "10 M");
	//file name by timestamp,eg as.log.20130227165009712
	fileChannel->setProperty(FileChannel::PROP_ARCHIVE, "timestamp");

	//reserver file timelength
	fileChannel->setProperty(FileChannel::PROP_PURGEAGE, "7 days");
	fileformatChannel->setChannel(fileChannel);

	if (m_logType & AlgorithmLogTypeFile)
	{
		m_pAsyncFile = new AsyncChannel(fileformatChannel);
		m_pSplitter->addChannel(m_pAsyncFile);
	}

	if (m_logType & AlgorithmLogTypeConsole)
	{
		m_pSplitter->addChannel(consoleFormatChannel);
	}

}

int AlgorithmLogger::analyseLogModules(std::string& logModules, std::map<string, int>& modulesMap)
{
	//去除所有空格
	string modulesAfterTrim = Poco::translate(logModules, " ", "");
	//string modulesAfterTrim = Poco::trim(logModules);
	StringTokenizer tokenizer(modulesAfterTrim, ",");

	int defaultLogLevel = 1;
	for (int i = 0; i < tokenizer.count(); i++)
	{
		string tmpLevel = tokenizer[i];
		StringTokenizer tmpTokenizer(tmpLevel, ":");
		if (tmpTokenizer.count() != 2)
		{
            printf("!!!!!!!!!!!!!!AlgorithmLog::createLogModule() failed[%s] is invalidate!!!!!!!!!!!!!!!!!!!!!! \n", tmpLevel.c_str());
			continue;			
		}
		else
		{
			string& moduleName = tmpTokenizer[0];
			int logLevle = 0;
			if (NumberParser::tryParse(tmpTokenizer[1], logLevle))
			{
				if (moduleName.compare("DEFAULT") == 0)
				{//默认日志级别
					defaultLogLevel = logLevle;
				}
				else
				{
					modulesMap[moduleName] = logLevle;
				}				
			}
			else
			{
                printf("!!!!!!!!!!!!!!AlgorithmLog::createLogModule() failed[%s] is invalidate!!!!!!!!!!!!!!!!!!!!!! \n", tmpLevel.c_str());
				continue;
			}
		}
	}

	return defaultLogLevel;
}

void AlgorithmLogger::createLogModule(std::string& logModules)
{
	std::map<string, int> logModuleMap;
	analyseLogModules(logModules, logModuleMap);

	std::map<string, int>::iterator logModuleIt = logModuleMap.begin();
	while (logModuleIt != logModuleMap.end())
	{
		format(logModuleIt->first, (Message::Priority)logModuleIt->second);
		logModuleIt++;
	}
}

bool AlgorithmLogger::init()
{
	m_pSplitter = new SplitterChannel;

	m_configFile = new File(m_configFilePath);

	bool configFileExist = false;
	try
	{
		configFileExist = m_configFile->exists();
	}
	catch (const Exception& e)
	{
		cout << "AlgorithmLog::init() failed  \n" << e.message() << " " << e.what() << endl;
	}

	if (!configFileExist)
	{
        cout << "AlgorithmLog::init() failed config file not find \n" << m_configFilePath << endl;
		return false;
	}

	string logFileName;
	string logLevels;

	INIConfigPtr configPtr = new IniFileConfiguration(m_configFilePath);
	
	try
	{
		logFileName = configPtr->getString("LogFileName");
		logLevels = configPtr->getString("LogLevel");
		m_logType = (AlgorithmLogType)configPtr->getInt("LogType");
	}
	catch (const std::exception& e)
	{
		cout << "AlgorithmLog::init() parse file failed " << e.what() << endl;
		return false;
	}

	if (logFileName.empty())
	{
		cout << "AlgorithmLog::init() failed config file not find " << logFileName << endl;
		return false;
	}

	//获取日志文件的绝对路径
	Path path(m_logFileDir);
	File file(path);
	if (!file.exists())
	{
		file.createDirectory();
	}
	string strBasePath = file.path();
	string strAbsolutePath = strBasePath.append("/").append(logFileName);
	createLogChannel(strAbsolutePath);


	//设置每个模块的级别
	if (logLevels.empty())
	{
        printf("!!!!!!!!!!!!!!AlgorithmLog::init() failed config file not find!!!!!!!!!!!!!!!!!!!!!! \n");
		return false;
	}

	//POCO日志支持多个模块使用不同的日志级别。
	//由于现在使用resip控制日志级别，暂时屏蔽掉相关业务逻辑
	createLogModule(logLevels);
	return true;

}

void AlgorithmLogger::run()
{
	if (!m_configFile->exists())
	{
		return;
	}

	Timestamp lastModifyTime = m_configFile->getLastModified();
	while (true)
	{
		//每秒检测一下配置文件是否变化了
		Thread::sleep(1000);
		if (m_configFile->exists())
		{
			if (m_configFile->getLastModified() != lastModifyTime)
			{
				lastModifyTime = m_configFile->getLastModified();

				//配置文件变化了，读取日志级别
				string logLevels;

				INIConfigPtr configPtr = new IniFileConfiguration(m_configFile->path());
				logLevels = configPtr->getString("LogLevel");

				if (logLevels.empty())
				{
					continue;
				}
				cout << "Log level change to " << logLevels << endl;

				std::map<string, int> logModuleMap;
				//分析日志级别
				int defaultLevel = analyseLogModules(logLevels, logModuleMap);

				for (auto logModule: logModuleMap)
				{
					if (m_objLoggerMap.count(logModule.first) > 0)
					{
						setLogLevel(m_objLoggerMap[logModule.first], logModule.second);
					}					
				}

			}
		}
	}
}

void AlgorithmLogger::setLogLevel(LoggerPtr p_pLog, int nLevel)
{
#if 1
	//1、flush状态enable
	FormattingChannel* pFChannel = static_cast<FormattingChannel*>(m_pAsyncFile->getChannel());
	pFChannel->getChannel()->setProperty("flush", "true");

	//2、写一条日志
	string strLog;
	if (nLevel == 0)
	{
		//0 means(turns off logging)
		p_pLog->error("AlgorithmLog:: Program Crash, Module[%s] Should Write All Cache Log!", p_pLog->name());
	}
	else
	{
		p_pLog->error("AlgorithmLog::run logLevle change to %?d", nLevel);

		//3、flush状态disable
		pFChannel->getChannel()->setProperty("flush", "false");
	}

	//4、设置日志级别
	p_pLog->setLevel(nLevel);
#endif
}

void AlgorithmLogger::log(AlgoLogPriority level, const string& msg)
{
	if (!msg.empty() && m_objLoggerMap.count(logModuleName) > 0)
	{
		switch (level)
		{
		case AlgoLogFatal:
			m_objLoggerMap[logModuleName]->fatal(msg);
			break;
		case AlgoLogCritical:
			m_objLoggerMap[logModuleName]->critical(msg);
			break;
		case AlgoLogError:
			m_objLoggerMap[logModuleName]->error(msg);
			break;
		case AlgoLogWarnning:
			m_objLoggerMap[logModuleName]->warning(msg);
			break;
		case AlgoLogNotice:
			m_objLoggerMap[logModuleName]->notice(msg);
			break;
		case AlgoLogInfo:
			m_objLoggerMap[logModuleName]->information(msg);
			break;
		case AlgoLogDebug:
			m_objLoggerMap[logModuleName]->debug(msg);
			break;
		case AlgoLogTrace:
			m_objLoggerMap[logModuleName]->trace(msg);
			break;
		default:
			break;
		}
		
	}
}


void AlgorithmLogger::log(AlgoLogPriority level, const char *fmt, ...)
{
	va_list pArg;

	va_start(pArg, fmt);
	char buf[4096] = {0};
	int n = vsprintf(buf, fmt, pArg);
	buf[4095] = '\0';
	va_end(pArg);

	gLogger->log( level, string(buf));
}


shared_ptr<AlgoLoggerInterface> AlgorithmLogger::algorithmLogerInit(const string& configFilePath, const string& logFileDir)
{
	//初始化poco日志
	if (!gLogger)
	{
		gLogger = std::make_shared<AlgorithmLogger>(configFilePath, logFileDir);
		gLogger->startWatch();
	}

	return gLogger;
}

//void log(AlgoLogPriority level, const string& msg);
//
//void log(AlgoLogPriority level, const char *fmt, ...);