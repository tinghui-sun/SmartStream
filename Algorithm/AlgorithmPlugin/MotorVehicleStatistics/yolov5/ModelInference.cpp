#include "modelInference.h"
//#include "yolov5.h"
#include <atomic> 
using namespace cv;

#ifndef WIN32
#define sprintf_s sprintf
#endif

ModelInference::ModelInference()
{
	string names = "人,自行车,汽车,摩托车,飞机,公共汽车,火车,卡车,船,红绿灯,消防栓,停车标志,停车费,长椅上,鸟,猫,狗,马,羊,牛,大象,熊,斑马,长颈鹿,背包,雨伞,手提包,领带,行李箱,飞盘,滑雪,滑雪板,体育球,风筝,棒球棒,棒球手套,滑板,冲浪板,网球拍,酒瓶,酒杯,杯,叉,刀,勺子,碗,香蕉,苹果,三明治,橙色,西兰花,胡萝卜,热狗,披萨,甜甜圈,蛋糕,椅子,沙发,盆栽植物,床,餐桌,厕所,电视,笔记本电脑,鼠标,遥控器,键盘,手机,微波炉,烤箱,烤面包机,水槽,冰箱,书,钟,花瓶,剪刀,泰迪熊,吹风机,牙刷";
	//std::string names = "person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush";
	vector<string> res;
	{
		std::istringstream tmp(names);
		std::string str;
		while (getline(tmp, str, ','))
			res.push_back(str);
	}
	m_className.clear();
	if (res.size() == 80)
	{
		for (auto i = 0; i < (int)res.size(); i++)
			m_className[i] = res[i];
	}
}

ModelInference::~ModelInference()
{
	LOGSTR(LDEBUG, "ModelInference::~ModelInference");
	m_mode.reset();
}

bool ModelInference::initialize(string configPath /*= ""*/, int gpuId /*= 0*/)
{
	cudaSetDevice(gpuId);
	LOG(LDEBUG, string("createInferBuilder success").c_str());
	m_mode.reset(new Yolo5());
	if (!m_mode) {
		LOG(LERROR, string("create mode failed").c_str());
		return false;
	}
	if (!m_mode->initialize(configPath.c_str())) {
		return false;
	}
	return true;
}

void ModelInference::runTest()
{
	DDIms dims;
	m_mode->getDDims(dims);

	int loop = 1000;
	vector<float> output(dims.outSize * dims.batchSize, 0);
	vector<float> input(dims.channel * dims.height * dims.width * dims.batchSize, 0.5f);
	TimeStatistics ts(FCL);
	for (auto i = 0; i < loop; i++)
		m_mode->inference(input.data(), output.data(), dims.batchSize);
}

bool ModelInference::run(string imgDir, Yolov5InferDirCallback callback, string tag /*= "default"*/)
{
	DDIms dims;
	m_mode->getDDims(dims);

	vector<float> output(dims.outSize * dims.batchSize, 0);
	auto pro = createBatchImageProcessor(imgDir);
	if (pro->initialize(dims.batchSize, dims.width, dims.height, dims.channel, true))
	{
		pro->resetIndex();
		float* data = nullptr;
		int batchSize = 0, dataSize = 0, count = 0;
		vector<string> files;
		vector<cv::Size> norSizes;
		vector<vector<Yolo::Detection>> yoloRes;
		TimeStatistics ts(FCL);

		while (true) {
			auto ret = pro->getData(&data, dataSize, batchSize, files, norSizes);
			if (batchSize <= 0)
				this_thread::yield();
			else
			{
				TimeStatistics ts(FCL);
				auto tm = chrono::steady_clock::now();
				m_mode->inference(data, output.data(), batchSize, tag);
				parserYoloResult(output, yoloRes);
				cout << "duration:" << chrono::duration<double, nano>(chrono::steady_clock::now() - tm).count() / 1e6 << endl;
				assert(files.size() == yoloRes.size() && files.size() == norSizes.size());
				if (callback)
				{
					vector<vector<Yolov5Detect>>_detect;
					for (auto i = 0;  i < (int)yoloRes.size() && i < (int)norSizes.size(); i++)
					{
						vector<Yolov5Detect> _detect1;
						for (auto j = 0; j < (int)yoloRes[i].size(); j++)
						{
							Yolov5Detect _detect2;
							_detect2.classId = yoloRes[i][j].class_id;
							_detect2.className = m_className.count(_detect2.classId) > 0 ? m_className[_detect2.classId] : "";
							_detect2.confidence = yoloRes[i][j].conf;
							_detect2.rect = rectRecover(norSizes[i].width, norSizes[i].height, dims.width, dims.height, yoloRes[i][j].bbox);
							_detect1.emplace_back(_detect2);
						}
						_detect.emplace_back(_detect1);
					}
					ret = callback(&files, &_detect);
				}
				/*if (true)
				{
					for (auto i = 0; i < (int)files.size(); i++)
					{
						Mat img = imread(files[i]);
						//auto img1 = imgResizeAndPadding(img, dims.width, dims.height);
						auto& res = yoloRes[i];
						for (size_t j = 0; j < res.size(); j++) 
						{
							//cv::Rect r = getRect(img1.cols, img1.rows, res[j].bbox, dims.width, dims.height);
							//cv::rectangle(img1, r, cv::Scalar(0x27, 0xC1, 0x36), 1);
							//cv::putText(img1, to_string((int)res[j].class_id) + "/" + to_string(res[j].conf).substr(0, 5), Point(r.x, r.y - 1), FONT_HERSHEY_PLAIN, 1.0, Scalar(0x00, 0x00, 0xff), 1);
							cv::Rect r = rectRecover(img.cols, img.rows, dims.width, dims.height, res[j].bbox);
							cv::rectangle(img, r, cv::Scalar(0x00, 0x00, 0xff), 1);
							cv::putText(img, to_string((int)res[j].class_id) + "/" + to_string(res[j].conf).substr(0, 5), Point(r.x, r.y - 1), FONT_HERSHEY_PLAIN, 1.0, Scalar(0x00, 0x00, 0xff), 1);
						}
						if (true)
							imwrite(files[i] + ".result.jpg", img);
					}
				}*/
				count += batchSize;
			}
			if (!ret)break;
		}
		LOGSTR(LDEBUG, "---count: " + to_string(count));
	}
	releaseBatchImageProcessor(pro);
	return true;
}

bool ModelInference::run(string file, vector<Yolov5Detect>& results, string tag /*= "default"*/)
{
	Mat img = imread(file);
	return run(img, results, tag);
}

bool ModelInference::run(Mat img, vector<Yolov5Detect>& results, string tag /*= "default"*/)
{
	results.clear();
	if (img.empty())
	{
		LOGSTR(LERROR, "img is empty!");
		return false;
	}
	//TimeStatistics ts(FCL);
	DDIms dims;
	m_mode->getDDims(dims);
	vector<float> imgData;
	if (imgConvert(img, dims.width, dims.height, dims.channel, imgData, true))
	{
		vector<float> output(dims.outSize * 1, 0);
		vector<vector<Yolo::Detection>> yoloRes;
		//auto tm = chrono::steady_clock::now();
		m_mode->inference(imgData.data(), output.data(), 1, tag);
		parserYoloResult(output, yoloRes);
		//cout << "duration:" << chrono::duration<double, nano>(chrono::steady_clock::now() - tm).count() / 1e6 << endl;
		assert(1 == yoloRes.size());
		for (auto &res : yoloRes[0])
		{
			Yolov5Detect detect;
			detect.classId = res.class_id;
			detect.className = m_className.count(detect.classId) > 0 ? m_className[detect.classId] : "";
			detect.confidence = res.conf;
			detect.rect = rectRecover(img.cols, img.rows, dims.width, dims.height, res.bbox);
			results.emplace_back(detect);
			//cv::rectangle(img, detect.rect, cv::Scalar(0x00, 0x00, 0xff), 1);
			//cv::putText(img, to_string(detect.classId) + "/" + to_string(detect.confidence).substr(0, 5), Point(detect.rect.x, detect.rect.y - 1), FONT_HERSHEY_PLAIN, 1.0, Scalar(0x00, 0x00, 0xff), 1);
		}
		return true;
	}
	return false;
}

bool ModelInference::run(vector<pair<string, Mat>>& imgs, vector<pair<int, vector<Yolov5Detect>>>& results, vector<int> &classFilter, pair<int, int> sizeFilter /*= make_pair(60, 60)*/,  string tag /*= "default"*/)
{
	results.clear();
	DDIms dims;
	m_mode->getDDims(dims);
	map<int, bool> _classFilter;
	for (auto classid : classFilter)
		_classFilter[classid] = true;

	vector<float> imgData, imgTmp;
	map<string, int> errNo;
	vector<string> succNo;
	for (auto& img : imgs)
	{
		if (img.second.empty())
		{
			LOGSTR(LWARN, "img[" + img.first+"] is empty!");
			errNo[img.first] = 1;
		}
		else 
		{
			if (imgConvert(img.second, dims.width, dims.height, dims.channel, imgTmp, true))
			{
				imgData.insert(imgData.end(), imgTmp.begin(), imgTmp.end());
				succNo.push_back(img.first);
			}
			else
			{
				LOGSTR(LWARN, "img[" + img.first + "] convert failure!");
				errNo[img.first] = 2;
			}
		}
	}

	int size = (int)succNo.size();
	vector<float> output(dims.outSize * size, 0);
	vector<vector<Yolo::Detection>> yoloRes;
	m_mode->inference(imgData.data(), output.data(), size, tag);
	parserYoloResult(output, yoloRes);
	assert(size == (int)yoloRes.size());

	for (int i = 0, j = 0; i < (int)imgs.size() && j < size; i++)
	{
		if (errNo.count(imgs[i].first) > 0)
		{
			results.emplace_back(make_pair(errNo[imgs[i].first], vector<Yolov5Detect>()));
		}
		else
		{
			auto _yoloRes = &yoloRes[j++];
			vector<Yolov5Detect> _result;
			for (int k = 0; k < (int)_yoloRes->size() ; k++)
			{
				Yolov5Detect detect;
				detect.classId = (*_yoloRes)[k].class_id;
				if(_classFilter.size() > 0 &&  _classFilter.count(detect.classId) <= 0) continue;
				detect.className = m_className.count(detect.classId) > 0 ? m_className[detect.classId] : "";
				detect.confidence = (*_yoloRes)[k].conf;
				detect.rect = rectRecover(imgs[i].second.cols, imgs[i].second.rows, dims.width, dims.height, (*_yoloRes)[k].bbox);
				if(detect.rect.width < sizeFilter.first && detect.rect.height < sizeFilter.second) continue;
				_result.emplace_back(detect);
				//cv::rectangle(img, detect.rect, cv::Scalar(0x00, 0x00, 0xff), 1);
				//cv::putText(img, to_string(detect.classId) + "/" + to_string(detect.confidence).substr(0, 5), Point(detect.rect.x, detect.rect.y - 1), FONT_HERSHEY_PLAIN, 1.0, Scalar(0x00, 0x00, 0xff), 1);
			}
			results.emplace_back(make_pair(0, _result));
		}
	}
	return true;
}

bool ModelInference::runStream(string url, StreamType type, Yolov5InferStreamCallback callback, int fps, string tag /*= "default"*/)
{
	if (type == StreamType::STFILE)
	{
		bool vaild = false;
		{
			VideoCapture cap(url);
			if((vaild = cap.isOpened()) == true)
				cap.release();
		}
		if (vaild)
		{
			list<Mat> cache;
			bool ret = true, thrdExit = false;
			mutex mtx;
			atomic_int cacheSize(0);
			//VideoCapture 不能在线程中运行，会报错
			VideoCapture capture(url);
			if (capture.isOpened())
			{
				thread([&]() {
					vector<Yolov5Detect> detect;
					char buf[128] = { 0 };
					Mat frame;
					while (ret && callback)
					{
						if (cacheSize <= 0)
							this_thread::yield();
						else
						{
							{
								unique_lock<mutex> loc(mtx);
								frame = cache.front();
								cache.pop_front();
								cacheSize--;
							}
							if (!frame.empty())
							{
								auto tm0 = chrono::steady_clock::now().time_since_epoch().count(), tm1 = tm0, tm2 = tm0;
								if (run(frame, detect, tag))
								{
									tm1 = chrono::steady_clock::now().time_since_epoch().count();
									for (auto& _detect : detect)
									{
										cv::rectangle(frame, _detect.rect, cv::Scalar(0x00, 0x00, 0xff), 1);
										cv::putText(frame, to_string(_detect.classId) + "/" + to_string(_detect.confidence).substr(0, 5), Point(_detect.rect.x, _detect.rect.y - 1), FONT_HERSHEY_PLAIN, 1.0, Scalar(0x00, 0x00, 0xff), 1);
									}
									tm2 = chrono::steady_clock::now().time_since_epoch().count();
									auto _ret = callback(&frame);
									ret &=_ret;
								} 
								auto tm3 = chrono::steady_clock::now().time_since_epoch().count();
								sprintf_s(buf, "infer-duration: infer[%0.4f]  draw[%0.4f]  callback[%0.4f]  -->%d  ---------->>total[%0.4f]", (tm1 - tm0) / 1e6, (tm2 - tm1) / 1e6, (tm3 - tm2) / 1e6, (int)detect.size(), (tm3 - tm0) / 1e6);
								LOG(LDEBUG, buf);
							}
						}
					}
					LOGSTR(LWARN, "runStream->infer_thread exit!!!");
					thrdExit = true;
				}).detach();
				Mat frame;
				auto frameDur = 1.0 * 1e9 / fps;
				int grabErrCount = 0;
				char buf[128] = { 0 };
				while (ret && callback)
				{
					if (cacheSize >= 2)
					{
						this_thread::yield();
						cout << "read frame -- yield: " << cacheSize << endl;
					}
					else
					{
						auto tm0 = chrono::steady_clock::now().time_since_epoch().count(), tm1 = tm0, tm2 = tm0, sleepTm = tm0, sp = tm0;
						if (capture.grab())
						{
							grabErrCount = 0;
							capture >> frame;
							if (frame.empty())
							{
								ret = false;
								LOGSTR(LWARN, "end of reading video file!!");
								break;
							}
							{
								unique_lock<mutex> loc(mtx);
								cache.push_back(frame);
								cacheSize++;
							}
							tm1 = chrono::steady_clock::now().time_since_epoch().count();
							sleepTm = frameDur - (tm1 - tm0);
							if (sleepTm > 0)
								this_thread::sleep_for(chrono::duration<double, nano>(sleepTm));
							sp = chrono::steady_clock::now().time_since_epoch().count();
						}
						else
						{
							grabErrCount += 1;
							if (grabErrCount >= 5)
							{
								ret = false;
								LOGSTR(LWARN, "grab frame LERROR more than 5!!!");
								break;
							}
						}
						tm2 = chrono::steady_clock::now().time_since_epoch().count();
						sprintf_s(buf, "capture_duration:%0.4f  ----->  capture:%0.4f sp:%0.4f[%0.4f] other:%0.4f", (tm2 - tm0) / 1e6, (tm1 - tm0) / 1e6, (sp - tm1) / 1e6, sleepTm / 1e6, (tm2 - sp) / 1e6);
						LOG(LDEBUG, buf);
					}
				}
				capture.release();
				while (!thrdExit)
					this_thread::yield();
			}
			else
				ret = false;
			return true;
		}
		LOGSTR(LERROR, "open " + url + " failure!");
		return false;
	}
	LOGSTR(LERROR, "unsupported stream type!");
	return false;
}

void ModelInference::parserYoloResult(vector<float>& data, vector<vector<Yolo::Detection>>& result)
{
	result.clear();
	DDIms dims;
	m_mode->getDDims(dims);
	assert((data.size() % dims.outSize) == 0);
	auto fcount = (int)(data.size() / dims.outSize);
	result.resize(fcount);
	for (auto i = 0; i < fcount; i++) {
		auto& res = result[i];
		//nms(res, &data[i * dims.outSize], dims.confThresh, dims.nmsThresh);
	}
}
