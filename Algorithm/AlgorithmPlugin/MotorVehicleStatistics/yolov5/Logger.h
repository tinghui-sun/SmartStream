#pragma once
#include <NvInfer.h>
#include <string>
#include <iostream>
#include <functional>
#include <assert.h>
#include <memory>

enum LEVEL
{
	LFATAL = 0,
	LERROR,
	LWARN,
	LINFO,
	LDEBUG
};

class Logger : public nvinfer1::ILogger
{
public:
	virtual ~Logger();
	static Logger* GetInstance();

public:
	void log(Severity severity, const char* msg);
	void log(LEVEL level, const char* msg);

	void bindLogWriter(std::function<void(int, const char*)> writer);

private:
	Logger();

	std::function<void(int, const char*)> m_writer;
	static std::unique_ptr<Logger> _logger;
};


inline nvinfer1::ILogger& GETLOGGER() {
	if (true)
		return *Logger::GetInstance();
	return (nvinfer1::ILogger&)*(nvinfer1::ILogger*)(nullptr);
}

#define LOG(severity, msg){ Logger::GetInstance()->log(severity, msg);}

#define LOGSTR(severity, msg){Logger::GetInstance()->log(severity, ((string)msg).c_str());}