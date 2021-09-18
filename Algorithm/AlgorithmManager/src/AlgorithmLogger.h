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
	AlgorithmLogTypeConsole = 1,   //��־���������̨
	AlgorithmLogTypeFile = 1 << 1,    //��־������ļ�
	AlgorithmLogTypeAll = AlgorithmLogTypeConsole | AlgorithmLogTypeFile,//ͬʱ�������̨���ļ�
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
	* ��ʼ����־
	* @moduleName : ������
	* @configFilePath : ���þ���·��
	* @logFileDir : ��־�ļ�д��Ŀ¼
	* @return 0 ��ʼ���ɹ��� ����ʧ��
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

	//��ʼ��������־ģ��
	//��ʽ��ģ������:��־����, ģ���� : ��־����,
	void createLogModule(std::string& logModules);

	//������־��ʽ����־����
	void format(std::string module, Message::Priority msgLevel = Message::PRIO_TRACE);

	//�����Ĭ����־���𣬷���Ĭ����־���𡣷��򷵻�0
	int analyseLogModules(std::string& logModules, std::map<string, int>& modulesMap);
	
	//����POCO����ģ����־
	void setLogLevel(LoggerPtr p_pLog, int nLevel);	

	void startWatch();

private:
	AutoPtr<SplitterChannel> m_pSplitter;
	AutoPtr<AsyncChannel> m_pAsyncConsole;
	AutoPtr<AsyncChannel> m_pAsyncFile;
	string m_configFilePath;
	string m_logFileDir;
	File* m_configFile = nullptr;
	AlgorithmLogType m_logType;	//1 �������̨�� 2 ����ļ�
	Thread m_objWatchThread;
	std::map<string, LoggerPtr> m_objLoggerMap;
};