#include "AlgorithmManager.h"
#include "Poco/Manifest.h"
#include "Poco/Exception.h"
#include "Poco/Path.h"
#include "Poco/File.h"
#include "Poco/AutoPtr.h"
#include "Poco/Exception.h"
#include "Poco/Environment.h"
#include "Poco/Util/IniFileConfiguration.h"
#include <iostream>
#include "AlgorithmLogger.h"

using Poco::AutoPtr;
using Poco::Path;
using Poco::File;
using Poco::Exception;
using Poco::ClassLoader;
using Poco::Manifest;
using Poco::SharedLibrary;
using Poco::AbstractMetaObject;
using Poco::NotFoundException;
using Poco::InvalidAccessException;
using Poco::Environment;
using Poco::Util::IniFileConfiguration;

string ALGOName[]
{
	"MultiTargetDetection",
	"FaceDetection",
	"FaceRecognition",
	"BodyRecognition",
	"BodyStatistics",
	"MotorVehicleRecognition",
	"MotorVehicleStatistics",
	"NoMotorVehicleRecognition",
	"SceneRecognition",
	"ThingRecognition",
	"PlateRecognition",
	"CommonRecognition",
	"VideoQualityDetection",
};

AlgorithmManager::AlgorithmManager()
{

}

AlgorithmManager::~AlgorithmManager()
{

}


AlgorithmManager& AlgorithmManager::instances()
{
	static AlgorithmManager devManager;
	return devManager;
}

bool AlgorithmManager::initManager()
{
	if (m_inited)
	{
		return true;
	}
	string appPath = Path::current();// = System::getApplicationPath();
	string pluginPath = appPath + "/plugins/";// = System::getApplicationPath();
	string configFilePath = pluginPath + "/plugins.ini";

	AlgorithmLogger::algorithmLogerInit(configFilePath, pluginPath);

	AutoPtr<IniFileConfiguration> configRead = new IniFileConfiguration();
	try
	{
		configRead->load(configFilePath);
	}
	catch (const Exception& e)
	{
		AlgorithmLogger::instance()->log(AlgoLogError, "AlgorithmManager::AlgorithmManager failed! " + e.message());
		return false;
	}

	string curPath;
	try
	{
#ifdef __GNUC__	
		curPath = Environment::get("LD_LIBRARY_PATH");
#else
		curPath = Environment::get("Path");
#endif
	}
	catch (const Exception& e)
	{
		AlgorithmLogger::instance()->log(AlgoLogError, "LD_LIBRARY_PATH or Path not found!!");
	}

	for (int i = 0; i < ALGOTypeMax; i++)
	{

		string pluginName = configRead->getString(ALGOName[i], "");
		if (!pluginName.empty())
		{
			if (!curPath.empty())
			{
#ifdef __GNUC__

				curPath.append(":");
#else	
				curPath.append(";");
#endif
			}

			//���λ�� Ӧ�ó���Ŀ¼/plugins/���Ŀ¼/��
			//��ͬ�㷨�Ĳ��Ŀ¼��plugins/plugins.ini�����ļ���
			//�㷨���Ŀ¼��������㷨����Ķ�̬��������һ��		
			string curPluginPath = pluginPath + pluginName + "/lib";
			curPath.append(curPluginPath); //�������Ŀ¼

			//string libPath = appPath + "/plugins/" + pluginName + "/" + pluginName;
			//libPath.append(SharedLibrary::suffix());
			//m_PluginPathMap[(ALGOType)i] = libPath;

			//��¼�����̬������			
#ifdef __GNUC__

			m_PluginPathMap[i] = curPluginPath + "/lib" + pluginName.append(SharedLibrary::suffix());
#else	
			m_PluginPathMap[i] = curPluginPath + "/" + pluginName.append(SharedLibrary::suffix());
#endif
			AlgoMsgError(AlgorithmLogger::instance(), "AlgorithmManager::AlgorithmManager add plugin %s", m_PluginPathMap[i].c_str());
		}
		else
		{
			AlgoMsgError(AlgorithmLogger::instance(), "AlgorithmManager::AlgorithmManager %s plugin not found", ALGOName[i].c_str());
		}
	}

	//�Ѷ�̬��λ�ü�¼��Path����������loadLibraryʱ���ڸ�Ŀ¼�²��Ҷ�̬��	
#ifdef __GNUC__		
	Environment::set("LD_LIBRARY_PATH", curPath);
#else	
	Environment::set("Path", curPath);
#endif
	m_inited = true;
	return m_inited;
}


AlgorithmPluginInterface* AlgorithmManager::getAlgorithmPlugin(ALGOType type, list<int> gpuNum)
{
	if (!m_inited)
	{
		if (!initManager())
		{
			cout << "AlgorithmManager::getAlgorithmPlugin init failed!" << endl;
			return nullptr;
		}
	}

	if (m_PluginPathMap.count(type)<=0 || m_PluginPathMap[type].empty())
	{		
		AlgorithmLogger::instance()->log(AlgoLogError, "AlgorithmManager::getAlgorithmPlugin plugin not exit! %d ", type);
		return nullptr;
	}

	AlgorithmPluginInterface* algorithmPlugin = nullptr;
	Poco::FastMutex::ScopedLock locker(m_AlgorithmPluginMapMutex);
	auto algorithmPluginIt = m_AlgorithmPluginMap.find(type);
	if (algorithmPluginIt == m_AlgorithmPluginMap.end())
	{//ÿ�����͵��㷨���ֻ��һ��
		algorithmPlugin = createAlgorithmPlugin(type, gpuNum);
		if (nullptr != algorithmPlugin)
		{
			m_AlgorithmPluginMap[type] = algorithmPlugin;
		}		
	}
	else
	{//�㷨����Ѽ��أ�ʹ���Ѽ��ص��㷨���
		algorithmPlugin = algorithmPluginIt->second;
	}

	return algorithmPlugin;
}


AlgorithmPluginInterface* AlgorithmManager::createAlgorithmPlugin(ALGOType type, list<int> gpuNum)
{
	string pluginPath = m_PluginPathMap[type];


	auto classLoaderIt = m_ClassLoaderMap.find(type);
	if (classLoaderIt == m_ClassLoaderMap.end())
	{//��Э���Ӧ��classLoader�����ڣ� ����һ��!
		m_ClassLoaderMap[type] = new ClassLoader<AlgorithmPluginInterface>();
	}

	ClassLoader<AlgorithmPluginInterface>* algorithmClassLoader = m_ClassLoaderMap[type];
	AlgorithmPluginInterface* algorithmPlugin = nullptr;
 	if (!algorithmClassLoader->isLibraryLoaded(pluginPath))
	{
		try
		{
			algorithmClassLoader->loadLibrary(pluginPath);
		}
		catch (const Exception& e)
		{
			AlgorithmLogger::instance()->log(AlgoLogError, "AlgorithmManager::createAlgorithmPlugin  load plugin lib failed! " + pluginPath + " " + e.message());
			return nullptr;
		}
	}

	try
	{
		PluginParam param;
		param.gpuList = gpuNum;
		param.pluginPath = Path(pluginPath).parent().parent().absolute().toString();
		param.logger = AlgorithmLogger::instance();
		
		//���������������AlgorithmPlugin
		algorithmPlugin = algorithmClassLoader->classFor("AlgorithmPlugin").create();
		//��ʼ�����
		if (algorithmPlugin->pluginInitialize(param) != ErrALGOSuccess)
		{
			AlgorithmLogger::instance()->log(AlgoLogError, "AlgorithmManager::createDeviceFactory pluginInitialize failed��");
			return nullptr;
		}
	}
	catch (...)
	{		
		AlgorithmLogger::instance()->log(AlgoLogError, "AlgorithmManager::createDeviceFactory  load class  AlgorithmPlugin failed! " + pluginPath);
		return nullptr;
	}

	return algorithmPlugin;
}

void AlgorithmManager::testLog()
{
	AlgorithmLogger::instance()->forceFlushLog();
}