#pragma once
#if defined(_WIN32)
#ifdef GLGOSDK_EXPORTS
#define GLGOSDK_API __declspec(dllexport)
#else
#define GLGOSDK_API __declspec(dllimport)
#if defined(_DEBUG)
#pragma comment(lib, "AlgorithmManagerd.lib")
#else
#pragma comment(lib, "AlgorithmManager.lib")
#endif
#endif
#else
#define GLGOSDK_API
#endif
#include "Algorithm.h"
#include "Poco/Mutex.h"
#include "Poco/ClassLoader.h"
#include <map>
#include <string>

using Poco::ClassLoader;
using std::map;
using std::string;
class GLGOSDK_API AlgorithmManager
{
public:
	~AlgorithmManager();

	static AlgorithmManager& instances();

	//���ݽ���Э������Id��ȡ��Ӧ��DeviceFactory���ⲿֻʹ�ò��ͷ�
	//�����ڲ��Ѿ�������AlgorithmPluginInterface::pluginInitialize��
	AlgorithmPluginInterface* getAlgorithmPlugin(ALGOType type);

	bool initManager();

	void testLog();

private:
	AlgorithmManager();
	AlgorithmPluginInterface* createAlgorithmPlugin(ALGOType type);

private:
	Poco::FastMutex m_AlgorithmPluginMapMutex;
	//key protocolName, value DeviceFactory
	map<int, string> m_PluginPathMap;
	map<int, AlgorithmPluginInterface*> m_AlgorithmPluginMap;
	map<int, ClassLoader<AlgorithmPluginInterface>*> m_ClassLoaderMap;
	bool m_inited = false;

};