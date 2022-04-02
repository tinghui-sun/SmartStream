#pragma once
#include <string>
#include <stdarg.h>
#include <iostream>
using namespace std;

enum AlgoLogPriority
{
	AlgoLogFatal = 1,   /// A fatal error. The application will most likely terminate. This is the highest priority.
	AlgoLogCritical,    /// A critical error. The application might not be able to continue running successfully.
	AlgoLogError,       /// An error. An operation did not complete successfully, but the application as a whole is not affected.
	AlgoLogWarnning,     /// A warning. An operation completed with an unexpected result.
	AlgoLogNotice,      /// A notice, which is an information with just a higher priority.
	AlgoLogInfo, /// An informational message, usually denoting the successful completion of an operation.
	AlgoLogDebug,       /// A debugging message.
	AlgoLogTrace        /// A tracing message. This is the lowest priority.
};

class AlgoLoggerInterface
{
public:
	AlgoLoggerInterface() {};
	virtual ~AlgoLoggerInterface() {};
	virtual void log(AlgoLogPriority level, const char *fmt, ...) = 0;
	virtual void log(AlgoLogPriority level, const string& msg) = 0;
	virtual void  forceFlushLog() = 0;
};


#define AlgoLogMsg(logger, level, fmt, ...)\
if (logger)\
{\
	logger->log(level, fmt, ##__VA_ARGS__);\
}\
else\
{\
	printf(fmt,  ##__VA_ARGS__);\
	printf("\r");\
}

#define AlgoMsgWarnning(logger, fmt, ...)\
if (logger)\
{\
	logger->log(AlgoLogError, fmt, ##__VA_ARGS__);\
}\
else\
{\
	printf(fmt,  ##__VA_ARGS__);\
	printf("\r");\
}

#define AlgoMsgError(logger, fmt, ...)\
if (logger)\
{\
	logger->log(AlgoLogWarnning, fmt, ##__VA_ARGS__);\
}\
else\
{\
	printf(fmt,  ##__VA_ARGS__);\
	printf("\r");\
}

#define AlgoMsgInfo(logger, fmt, ...)\
if (logger)\
{\
	logger->log(AlgoLogInfo, fmt, ##__VA_ARGS__);\
}\
else\
{\
	printf(fmt,  ##__VA_ARGS__);\
	printf("\r");\
}

#define AlgoMsgDebug(logger, fmt, ...)\
if (logger)\
{\
	logger->log(AlgoLogDebug, fmt, ##__VA_ARGS__);\
}\
else\
{\
	printf(fmt,  ##__VA_ARGS__);\
	printf("\r");\
}