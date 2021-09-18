#include <numeric>
#include <fstream>
#include "BaseModel.h"
#include "common.h"
#include "utils.h"
#include "Logger.h"
#include "calibrator.h"

BaseModel::BaseModel(int inputWidth, int inputHeight, int inputChannels, int outputSize):m_inputWidht(inputWidth), m_inputHight(inputHeight),m_inputChannels(inputChannels),m_outputSize(outputSize)
{
}

BaseModel::~BaseModel()
{
	LOGSTR(LDEBUG, "BaseModel::~BaseModel");
}

bool BaseModel::initialize(const char* configPath)
{
	auto cfg = getCfg(configPath);
	m_maxBatchSize = cfg["maxBatchSize"];
	/*m_params.dataDir = cfg["dataDir"];
	m_params.engineFileName = buildPath(cfg["engineFileName"], m_params.dataDir);
	m_params.batchSize = max(1, min((int)cfg["run"]["batchSize"], (int)cfg["maxBatchSize"]));
	m_params.confThresh = cfg["run"]["confThresh"];
	m_params.nmsThresh = cfg["run"]["nmsThresh"];
	m_params.dlaCore = 0;
	m_params.int8 = false;
	LOG(LDEBUG, string("init engine...").c_str());
	if (!deserializeEngine())
	{
		m_params.batchSize = max(1, (int)cfg["maxBatchSize"]);
		m_params.weightsFileName = cfg["build"]["weightsFileName"];
		m_params.dlaCore = cfg["build"]["dlaCore"];
		m_params.int8 = cfg["build"]["useInt8"];
		if (m_params.int8)
		{
			m_params.calibrationFileName = cfg["build"]["calibrationFileName"];
			m_params.calibrationImgFolder = cfg["build"]["calibrationImgFolder"];
			m_params.calibrationImgBatchs = cfg["build"]["calibrationImgBatchs"];
		}
		m_params.modeType = cfg["build"]["modeType"];
		LOGSTR(LDEBUG, "BaseModel:" + m_params.to_string());
		if (!initializeEngine())
			return false;
	}
	else {
		LOGSTR(LDEBUG, m_params.to_string());
	}

	m_int8calibrator.reset();
	m_builder.reset();
	m_runtime.reset();
	m_contextCleanTimer.reset(new TinyTimer);
	std::function<void(void)> checkFunc = std::bind(&BaseModel::checkOuttimeContext, this);
	m_contextCleanTimer->AsyncLoopExecute(60 * 1000, checkFunc);*/
	return true;
}

void BaseModel::inference(float* input, float* output, int batchSize, const std::string tag /*= "default"*/, bool allocMaxBatch /*= false*/)
{
	initCudaResource(tag, allocMaxBatch ? m_maxBatchSize : batchSize);

	auto context = m_engineContexts[tag];
	CHECK(cudaMemcpyAsync(context->cudaBuffers[context->inputBindingIndex], input, batchSize * m_inputChannels * m_inputHight * m_inputWidht * sizeof(float), cudaMemcpyHostToDevice, context->cudaStream));
	context->context->enqueue(batchSize, context->cudaBuffers.data(), context->cudaStream, nullptr);
	CHECK(cudaMemcpyAsync(output, context->cudaBuffers[context->outputBindingIndex], batchSize * m_outputSize * sizeof(float), cudaMemcpyDeviceToHost, context->cudaStream));
	cudaStreamSynchronize(context->cudaStream);
}

void BaseModel::getDDims(DDIms& dims)
{
	dims.batchSize = m_params.batchSize;
	dims.channel = m_inputChannels;
	dims.width = m_inputWidht;
	dims.height = m_inputHight;
	dims.outSize = m_outputSize;
	dims.confThresh = m_params.confThresh;
	dims.nmsThresh = m_params.nmsThresh;
}

void BaseModel::destroy()
{
	LOGSTR(LDEBUG, getClassName()+":BaseModel::destroy");
	m_contextCleanTimer.reset();
	m_int8calibrator.reset();
	{
		lock_guard<mutex> lg(m_engineContextsMtu);
		m_engineContexts.clear();
	}
	m_engine.reset();
	m_builder.reset();
	m_runtime.reset();
	delete this;
}

bool BaseModel::initializeEngine()
{
	map<string, Weights> weightMap;
	//加载权重文件
	weightMap = loadWeights(buildPath(m_params.weightsFileName, m_params.dataDir));
	if (weightMap.empty())
		return false;

	m_builder.reset(createInferBuilder(GETLOGGER()));
	if (!m_builder) {
		LOG(LERROR, string("createInferBuilder failed").c_str());
		return false;
	}

	LOG(LDEBUG, string("createInferBuilder success").c_str());
	const auto explicitBatch = 0U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	UniquePtr<INetworkDefinition> network(m_builder->createNetworkV2(explicitBatch));
	if (!network) {
		LOG(LERROR, string("createNetworkV2 failed").c_str());
		return false;
	}
	LOG(LDEBUG, string("createNetworkV2 success").c_str());
	UniquePtr<IBuilderConfig> config(m_builder->createBuilderConfig());
	if (!config) {
		LOG(LERROR, string("createBuilderConfig failed").c_str());
		return false;
	}
	LOG(LDEBUG, string("createBuilderConfig success").c_str());

	if (!createNetwork(network.get(), weightMap))
		return false;

	LOG(LDEBUG, string("createNetwork success").c_str());

	m_builder->setMaxBatchSize(m_params.batchSize);
	config->setMaxWorkspaceSize(16 * (1 << 20)); //16MB
	//config->setFlag(BuilderFlag::kLDEBUG);
	if (m_params.int8 && m_builder->platformHasFastInt8())
	{
		LOG(LWARN, "the platform support INT8 mode,  use INT8 mode by default");
		config->setFlag(BuilderFlag::kINT8);
		m_int8calibrator.reset(
			new Int8EntropyCalibrator2(
				m_params.batchSize,
				m_inputWidht, 
				m_inputHight, 				
				buildPath(m_params.calibrationFileName, m_params.dataDir).c_str(),
				buildPath(m_params.calibrationImgFolder, m_params.dataDir).c_str(),
				INPUT_BLOB_NAME.c_str()));

		config->setInt8Calibrator(m_int8calibrator.get());
	}
	else if (m_builder->platformHasFastFp16())
		config->setFlag(BuilderFlag::kFP16);
	if (m_params.dlaCore >= 0 && m_builder->getNbDLACores() > 0)
	{
		auto num = m_builder->getNbDLACores();
		if (m_params.dlaCore > num - 1) m_params.dlaCore = 0;
		LOGSTR(LWARN, "the platform have " + to_string(num) + " DLACore,  use index(" + to_string(m_params.dlaCore) + ")");
		config->setFlag(BuilderFlag::kGPU_FALLBACK);
		config->setDLACore(m_params.dlaCore);
		config->setDefaultDeviceType(DeviceType::kDLA);
		config->setFlag(BuilderFlag::kSTRICT_TYPES);
	}

	m_engine.reset(m_builder->buildEngineWithConfig(*network.get(), *config.get()));
	if (!m_engine) {
		LOG(LERROR, string("buildEngineWithConfig failed").c_str());
		return false;
	}
	network.reset();
	config.reset();
	for (auto& mem : weightMap) {
		delete[](float*)mem.second.values;
		mem.second.values = nullptr;
	}
	serializeEngine();
	return true;
}

bool BaseModel::deserializeEngine()
{
	if (m_params.engineFileName.length() > 0)
	{
		ifstream file(m_params.engineFileName, ios::binary);
		if (file.is_open())
		{
			file.seekg(0, file.end);
			auto size = file.tellg();
			file.seekg(0, file.beg);

			auto modelStream = new char[size];
			assert(modelStream);
			file.read(modelStream, size);
			file.close();

			m_runtime.reset(createInferRuntime(GETLOGGER()));
			if (!m_runtime) 
			{
				LOG(LERROR, string("createInferRuntime failed").c_str());
				return false;
			}
			LOG(LDEBUG, string("createInferRuntime success").c_str());
			m_engine.reset(m_runtime->deserializeCudaEngine(modelStream, size, nullptr));
			if (!m_engine)
			{
				LOG(LERROR, string("deserializeCudaEngine failed").c_str());
				return false;
			}
			LOG(LDEBUG, string("deserializeCudaEngine success").c_str());
			delete[]modelStream;
			modelStream = nullptr;
			return true;
		}
		else
			LOG(LERROR, string("deserialize engine failed, open file  LERROR").c_str());
	}
	else
		LOG(LERROR, string("deserialize engine failed, filename is empty").c_str());
	return false;
}

void BaseModel::serializeEngine()
{
	assert(m_engine);
	if (m_params.engineFileName.length() > 0)
	{
		UniquePtr<IHostMemory> modelStream(m_engine->serialize());
		assert(modelStream);
		ofstream file(m_params.engineFileName, ios::binary);
		if (file.is_open())
		{
			file.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
			file.close();
		}
		else
			LOG(LDEBUG, string("serialize engine failed, open file  LERROR").c_str());
		modelStream.reset();
	}
	else {
		LOG(LDEBUG, string("serialize engine failed, filename is empty").c_str());
	}
}

bool BaseModel::initCudaResource(string tag, int batchSize)
{
	lock_guard<mutex> lg(m_engineContextsMtu);
	if (m_engineContexts.count(tag) > 0)
	{
		auto context = m_engineContexts[tag];
		if (context->batchSize < batchSize)
			m_engineContexts.erase(tag);
		else
		{
			context->lastTime = chrono::steady_clock().now().time_since_epoch().count();
			return true;
		}
	}

	if (m_engineContexts.count(tag) <= 0)
	{
		shared_ptr<EngineContext> context = make_shared<EngineContext>();
		context->context = m_engine->createExecutionContext();
		context->batchSize = batchSize;
		auto bindings = m_engine->getNbBindings();
		context->inputBindingIndex = m_engine->getBindingIndex(INPUT_BLOB_NAME.c_str());
		context->outputBindingIndex = m_engine->getBindingIndex(OUTPUT_BLOB_NAME.c_str());

		assert(bindings == 2);
		context->cudaBuffers.resize(bindings);

		CHECK(cudaMalloc(&context->cudaBuffers[context->inputBindingIndex], batchSize * m_inputChannels * m_inputHight * m_inputWidht * sizeof(float)));
		CHECK(cudaMalloc(&context->cudaBuffers[context->outputBindingIndex], batchSize * m_outputSize * sizeof(float)));
		CHECK(cudaStreamCreate(&context->cudaStream));
		context->lastTime = chrono::steady_clock().now().time_since_epoch().count();
		m_engineContexts[tag] = context;
		LOGSTR(LINFO, getClassName()+":BaseModel::initCudaResource add tag:" + tag);
	}
	return true;
}

void BaseModel::checkOuttimeContext()
{
	auto curTm = chrono::steady_clock().now().time_since_epoch().count();
	lock_guard<mutex> lg(m_engineContextsMtu);
	for (auto itor = m_engineContexts.begin(); itor != m_engineContexts.end();)
	{
		if (curTm - itor->second->lastTime > 60 * 1000 * 1e6)
		{
			LOGSTR(LINFO, getClassName()+":BaseModel::checkOuttimeContext release tag:" + itor->first);
			itor = m_engineContexts.erase(itor);
		}
		else
			itor++;
	}
}