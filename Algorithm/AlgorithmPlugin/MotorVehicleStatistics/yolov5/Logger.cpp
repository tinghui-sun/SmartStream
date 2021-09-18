#include "logger.h"
#include <iostream>
#include<mutex>

std::unique_ptr<Logger> Logger::_logger;

Logger* Logger::GetInstance() {
	static std::once_flag _flag;
	std::call_once(_flag, [&]() {
		_logger.reset(new Logger());
	});
	return _logger.get();
}

Logger::Logger()
{
}

Logger::~Logger()
{
}

void Logger::bindLogWriter(std::function<void(int, const char*)> writer)
{
	m_writer = std::forward<std::function<void(int, const char*)>>(writer);
}

inline std::string getCurrentTm()
{
	auto now = std::chrono::system_clock::now();
	auto ttm = std::chrono::system_clock::to_time_t(now);
	struct tm* ptm = localtime(&ttm);
	char date[60] = { 0 };
	sprintf(date, "%d-%02d-%02d %02d:%02d:%02d.%.03f ", ptm->tm_year + 1900, ptm->tm_mon + 1, ptm->tm_mday,
		ptm->tm_hour, ptm->tm_min, ptm->tm_sec, (now.time_since_epoch().count() % 1000000000) / 1e6);
	return date;
}

void Logger::log(Severity severity, const char* msg)
{
	switch (severity)
	{
	case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
		log(LEVEL::FATAL, msg);
		break;
	case nvinfer1::ILogger::Severity::kERROR:
		log(LEVEL::ERROR, msg);
		break;
	case nvinfer1::ILogger::Severity::kWARNING:
		log(LEVEL::WARN, msg);
		break;
	case nvinfer1::ILogger::Severity::kINFO:
		log(LEVEL::INFO, msg);
		break;
	case nvinfer1::ILogger::Severity::kVERBOSE:
		log(LEVEL::DEBUG, msg);
		break;
	default:
		log(LEVEL::DEBUG, msg);
		break;
	}
}

void Logger::log(LEVEL level, const char* msg)
{
	if (m_writer)
	{
		switch (level)
		{
		case FATAL:
		case ERROR:
		case WARN:
		case INFO:
		case DEBUG:
			m_writer(level, msg);
			break;
		default:
			m_writer(LEVEL::DEBUG, msg);
			break;
		}
	}
	else
	{
		switch (level)
		{
		case LEVEL::FATAL: std::cout << getCurrentTm() << "[F] " << msg << std::endl; break;
		case LEVEL::ERROR: std::cout << getCurrentTm() << "[E] " << msg << std::endl; break;
		case LEVEL::WARN: std::cout << getCurrentTm() << "[W] " << msg << std::endl; break;
		case LEVEL::INFO: std::cout << getCurrentTm() << "[I] " << msg << std::endl; break;
		case LEVEL::DEBUG: std::cout << getCurrentTm() << "[D] " << msg << std::endl; break;
		default: std::cout << getCurrentTm() << "[D] " << msg << std::endl; break;
		}
	}
}