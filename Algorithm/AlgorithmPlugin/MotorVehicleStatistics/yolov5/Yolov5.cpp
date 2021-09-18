#include "Yolov5.h"
#include "common.h"
#include "utils.h"
#include <numeric>
#include <fstream>

const static int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;
const static int INPUT_WIDTH = Yolo::INPUT_W;
const static int INPUT_HEIGHT = Yolo::INPUT_H;
const static int INPUT_CHANNEL = Yolo::INPUT_C;

static int get_width(int x, float gw, int divisor = 8) {
	return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
	if (x == 1) return 1;
	int r = round(x * gd);
	if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
		--r;
	}
	return std::max<int>(r, 1);
}

Yolo5::Yolo5():BaseModel(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL, OUTPUT_SIZE)
{
}

Yolo5::~Yolo5()
{
	LOGSTR(LDEBUG, "Yolo5::~Yolo5");
}

bool Yolo5::initialize(const char* configPath)
{
	BaseModel::initialize(configPath);
	auto cfg = getCfg(configPath);
	m_params.dataDir = cfg["dataDir"];
	m_params.engineFileName =  buildPath(cfg["engineFileName"], m_params.dataDir);
	m_params.batchSize = max(1, min((int)cfg["run"]["batchSize"], (int)cfg["maxBatchSize"]));
	m_params.confThresh = cfg["run"]["confThresh"];
	m_params.nmsThresh = cfg["run"]["nmsThresh"];
	m_params.dlaCore = 0;
	m_params.int8 = false;
	LOG(LDEBUG, string("Yolo5 init engine...").c_str());
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
		LOGSTR(LDEBUG, "Yolov5:" + m_params.to_string());
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
	std::function<void(void)> checkFunc = std::bind(&Yolo5::checkOuttimeContext, this);
	m_contextCleanTimer->AsyncLoopExecute(60 * 1000, checkFunc);
	return true;
}



///-----------------------模型构造过程---------------------/////
bool Yolo5::createNetwork(INetworkDefinition* network, map<string, Weights>& weightMap)
{
	LOGSTR(LDEBUG, "createNetwork ...");
	bool is_p6 = false, float gd = 0.33, float gw = 0.50;
	DataType dt = DataType::kFLOAT;

	auto net = std::string(m_params.modeType);
	if (net[0] == 's') {
		gd = 0.33;
		gw = 0.50;
	}
	else if (net[0] == 'm') {
		gd = 0.67;
		gw = 0.75;
	}
	else if (net[0] == 'l') {
		gd = 1.0;
		gw = 1.0;
	}
	else if (net[0] == 'x') {
		gd = 1.33;
		gw = 1.25;
	}
	//else if (net[0] == 'c' && argc == 7) {
	//	gd = atof(argv[5]);
	//	gw = atof(argv[6]);
	//}
	else {
		return false;
	}
	if (net.size() == 2 && net[1] == '6') {
		is_p6 = true;
	}

	if (is_p6)
	{
		createNetworkNormalP6(network, weightMap, dt, gd, gw);
	}
	else
	{
		createNetworkNormal(network, weightMap, dt, gd, gw);
	}

	return true;
}

void Yolo5::createNetworkNormal(INetworkDefinition* network, map<string, Weights>& weightMap, DataType dt, float& gd, float& gw)
{
	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME.c_str(), dt, Dims3{ 3, INPUT_HEIGHT, INPUT_WIDTH });
	assert(data);

	/* ------ yolov5 backbone------ */
	auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
	auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
	auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
	auto spp8 = SPP(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, 9, 13, "model.8");

	/* ------ yolov5 head ------ */
	auto bottleneck_csp9 = C3(network, weightMap, *spp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.9");
	auto conv10 = convBlock(network, weightMap, *bottleneck_csp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

	auto upsample11 = network->addResize(*conv10->getOutput(0));
	assert(upsample11);
	upsample11->setResizeMode(ResizeMode::kNEAREST);
	upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

	ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	auto cat12 = network->addConcatenation(inputTensors12, 2);
	auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
	auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

	auto upsample15 = network->addResize(*conv14->getOutput(0));
	assert(upsample15);
	upsample15->setResizeMode(ResizeMode::kNEAREST);
	upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

	ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
	auto cat16 = network->addConcatenation(inputTensors16, 2);

	auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

	/* ------ detect ------ */
	IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
	ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
	auto cat19 = network->addConcatenation(inputTensors19, 2);
	auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
	IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
	ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
	auto cat22 = network->addConcatenation(inputTensors22, 2);
	auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
	IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

	auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
	yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME.c_str());
	network->markOutput(*yolo->getOutput(0));
}

void Yolo5::createNetworkNormalP6(INetworkDefinition* network, map<string, Weights>& weightMap, DataType dt, float& gd, float& gw)
{
	//INetworkDefinition* network = builder->createNetworkV2(0U);

	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME.c_str(), dt, Dims3{ 3, INPUT_HEIGHT, INPUT_WIDTH });
	assert(data);

	//std::map<std::string, Weights> weightMap = loadWeights(wts_name);

	/* ------ yolov5 backbone------ */
	auto focus0 = focus(network, weightMap, *data, 3, get_width(64, gw), 3, "model.0");
	auto conv1 = convBlock(network, weightMap, *focus0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(9, gd), true, 1, 0.5, "model.4");
	auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
	auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
	auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
	auto spp10 = SPP(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), 3, 5, 7, "model.10");
	auto c3_11 = C3(network, weightMap, *spp10->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.11");

	/* ------ yolov5 head ------ */
	auto conv12 = convBlock(network, weightMap, *c3_11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
	auto upsample13 = network->addResize(*conv12->getOutput(0));
	assert(upsample13);
	upsample13->setResizeMode(ResizeMode::kNEAREST);
	upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
	ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
	auto cat14 = network->addConcatenation(inputTensors14, 2);
	auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

	auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
	auto upsample17 = network->addResize(*conv16->getOutput(0));
	assert(upsample17);
	upsample17->setResizeMode(ResizeMode::kNEAREST);
	upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
	ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
	auto cat18 = network->addConcatenation(inputTensors18, 2);
	auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

	auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
	auto upsample21 = network->addResize(*conv20->getOutput(0));
	assert(upsample21);
	upsample21->setResizeMode(ResizeMode::kNEAREST);
	upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
	ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
	auto cat22 = network->addConcatenation(inputTensors21, 2);
	auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

	auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
	ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
	auto cat25 = network->addConcatenation(inputTensors25, 2);
	auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

	auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
	ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
	auto cat28 = network->addConcatenation(inputTensors28, 2);
	auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

	auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
	ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
	auto cat31 = network->addConcatenation(inputTensors31, 2);
	auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

	/* ------ detect ------ */
	IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
	IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
	IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
	IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

	auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
	yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME.c_str());
	network->markOutput(*yolo->getOutput(0));
}
