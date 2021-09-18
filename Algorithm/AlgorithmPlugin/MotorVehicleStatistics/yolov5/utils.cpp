#include "utils.h"
#include "logger.h"
#include <thread>
#include <numeric>
#include <stdarg.h>
#include <iostream>
#include <fstream>
#include <math.h>
#ifdef WIN32
#include <filesystem>
#else
#include <sys/stat.h>
#include <dirent.h>
#endif

using namespace cv;


const json& getCfg(const char* config) {
	static json cfg;
	static std::once_flag occfg;
	std::call_once(occfg, [&]() {
		try
		{
			string strconfig = config != nullptr && strlen(config) > 0 ? config : "config.cfg";
			LOGSTR(LINFO, "loading config[" + strconfig +"]...");
			string content = "";
			{
				std::ifstream cfg(strconfig);
				if (cfg.good()) {
					std::stringstream buffer;
					buffer << cfg.rdbuf();
					content = string(buffer.str());
				}
			}
			cfg = json::parse(content);
		}
		catch (...)
		{
			LOGSTR(LERROR, "parse config failed, use default cfg");
			cfg["maxBatchSize"] = 1;
			cfg["dataDir"] = "./";
			cfg["engineFileName"] = "yolov5.engine";
			cfg["build"]["dlaCore"] = 0;
			cfg["build"]["useInt8"] = false;
			cfg["build"]["modeType"] = "s";
			cfg["build"]["weightsFileName"] = "yolov5.wts";
			cfg["build"]["calibrationFileName"] = "";
			cfg["build"]["calibrationImgFolder"] = "";
			cfg["build"]["calibrationImgBatchs"] = 0;
			cfg["run"]["batchSize"] = 1;
			cfg["run"]["confThresh"] = 0.5f;
			cfg["run"]["nmsThresh"] = 0.4f;

			cfg["tracker"]["minHits"] = 3;
			cfg["tracker"]["maxAge"] = 2;
			cfg["tracker"]["iouThreshold"] = 0.3f;
			LOGSTR(LINFO, "default-cfg:\n" + cfg.dump(2));
		}
	});
	return cfg;
}

Mat imgResizeAndPadding(Mat& img, int width, int height) 
{
	int w, h, x, y;
	float r_w = width / (img.cols*1.0);
	float r_h = height / (img.rows*1.0);
	if (r_h > r_w) {
		w = width;
		h = r_w * img.rows;
		x = 0;
		y = (height - h) / 2;
	}
	else {
		w = r_h * img.cols;
		h = height;
		x = (width - w) / 2;
		y = 0;
	}
	Mat re(h, w, CV_8UC3);
	resize(img, re, re.size(), 0, 0, INTER_LINEAR);
	Mat out(height, width, CV_8UC3, Scalar(128, 128, 128));
	re.copyTo(out(Rect(x, y, re.cols, re.rows)));
	return out;
}

bool imgConvert(string file, int width, int height, int channel, vector<float>& result, bool padding, cv::Size* norSize)
{
	Mat img = imread(file, 1);
	if (img.empty())
	{
		LOGSTR(LERROR, "imread " + file + " failed");
		return false;
	}
	return imgConvert(img, width, height, channel, result, padding, norSize);
}

bool imgConvert(Mat img, int width, int height, int channel, vector<float>& result, bool padding, cv::Size* norSize)
{
	//tensorRT数据格式为NCHW 通道为RGB 数值范围【0， 1】数据类型为float， opencv为HWC，BGR，【0->255】uchar
	if (norSize) 
	{
		(*norSize).width = img.cols;
		(*norSize).height = img.rows;
	}
	if (padding)
		img = imgResizeAndPadding(img, width, height);
	else
		resize(img, img, Size(width, height));
	int img_channel = img.channels();
	if (channel == 3 && img_channel == 1)
		cvtColor(img, img, COLOR_GRAY2RGB);
	else if(channel == 3 && img_channel > 1)
		cvtColor(img, img, COLOR_BGR2RGB);
	else if (channel == 1 && img_channel == 1) {}
	else if(channel == 1 && img_channel > 1)
		cvtColor(img, img, COLOR_BGR2GRAY);
	else
	{
		LOGSTR(LERROR,  "unsupported channels");
		return false;
	}
	//uchar转float并缩放到0->1
	img.convertTo(img, CV_32FC3, 1. / 255.);

	//拆分通道
	vector<Mat> input_channels(channel);
	split(img, input_channels);

	//归一化
	result.resize(height * width * channel);
	auto data = result.data();
	auto channel_length = height * width;
	for (auto i = 0; i < channel; ++i) 
		memcpy(data + i * channel_length, input_channels[i].data, channel_length * sizeof(float));
	return true;
}

#ifndef WIN32
inline void listFilesUnix(const string& path, vector<string>& files)
{
	DIR *dir = nullptr;
	struct dirent *ptr = nullptr;
	if ((dir = opendir(path.c_str())) == nullptr)
	{
		LOGSTR(LERROR, path + " not exist!!");
		return;
	}
	while ((ptr = readdir(dir)) != nullptr)
	{
		if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0)
			continue;
		else if (ptr->d_type == 8)    //file  
			files.push_back(path + "/" + string(ptr->d_name));
		else if (ptr->d_type == 10)    //link file  
			continue;
		else if (ptr->d_type == 4)    //dir  
		{
			string _path = path + "/" + string(ptr->d_name);
			listFilesUnix(_path, files);
		}
	}
	closedir(dir);
}
#endif

//bool listFiles(const string& path, vector<string>& files)
//{
//	LOGSTR(LDEBUG, "listFiles enter");
//#ifdef WIN32
//	auto _path = filesystem::path(path);
//	if (!exists(_path))
//	{
//		LOGSTR(LERROR, path + " not exist!!");
//		return false;
//	}
//	auto begin = filesystem::recursive_directory_iterator(_path);
//	auto end = filesystem::recursive_directory_iterator();
//	for (auto it = begin; it != end; it++) {
//		if (filesystem::is_regular_file(*it)) 
//			files.emplace_back(it->path().generic_string());
//	}
//#else
//	listFilesUnix(path, files);
//#endif
//	LOGSTR(LDEBUG, "listFiles leave, num: " + to_string(files.size()));
//	return true;
//}

string locateFile(string fileName, vector<string>& dirs)
{
	string filepath = "";
	for (auto& dir : dirs) {
		filepath.clear();
		if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
			filepath = dir + "/" + fileName;
		else
			filepath = dir + fileName;
		ifstream checkFile(filepath);
		if (checkFile.is_open()) {
			checkFile.close();
			break;
		}
	}
	if (filepath.empty()) {
		string dirList = accumulate(dirs.begin() + 1, dirs.end(), dirs.front(),
			[](const string& a, const string& b) { return a + "\n\t" + b; });
		string output = "could not find " + fileName + " in " + dirList;
		LOG(LERROR, output.c_str());
	}
	return filepath;
}

string locateFolder(string folderName, vector<string>& dirs)
{
	string folderpath = "";
	for (auto& dir : dirs) {
		folderpath.clear();
		if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
			folderpath = dir + "/" + folderName;
		else
			folderpath = dir + folderName;
		//if (testFileExist(folderpath))
		//	break;
	}
	if (folderpath.empty()) {
		string dirList = accumulate(dirs.begin() + 1, dirs.end(), dirs.front(),
			[](const string& a, const string& b) { return a + "\n\t" + b; });
		string output = "could not find " + folderName + " in " + dirList;
		LOG(LERROR, output.c_str());
	}
	return folderpath;
}

string buildPath(string fileName, string dir)
{
	if (!dir.empty() && dir.back() != '/' && dir.back() != '\\')
		return dir + "/" + fileName;
	return dir + fileName;
}

//bool testFileExist(string file)
//{
//#ifdef WIN32
//	return filesystem::exists(file);
//#else
//	struct stat buffer;
//	return stat(file.c_str(), &buffer) == 0;
//#endif
//}

int getCpus()
{
	return (int)std::thread::hardware_concurrency();
}

void createTestJpg(int value, float inc, string path, int num)
{
	for (auto i = 0; i < num; i++)
	{
		int va = value + int(inc * (float)i);
		Mat img = cv::Mat(100, 100, CV_8UC3, Scalar(va, va, va));
		imwrite(path + "/image" + to_string(i+1) + ".jpg", img);
	}
}

template <class Type>
Type stringToNum(const string& str)
{
	istringstream iss(str);
	Type num;
	iss >> num;
	return num;
}

vector<float> characteristicFile2vec(const string& filename)
{
	//vector<string> data;
	vector<float> buf;
	vector<char> ouput;
	ifstream file(filename);
	if (file.good())
	{
		//file >> noskipws;
		copy(istream_iterator<char>(file), istream_iterator<char>(), back_inserter(ouput));
	}
	string tmp = "";
	for (size_t i = 0, j = ouput.size() - 1; i < ouput.size(); i++)
	{
		char& c = ouput[i];
		if (c != '\r' && c != '\n' && c != ',' && c != ' ') {
			tmp += c;
			if (i == j)
			{
				//data.push_back(tmp);
				buf.push_back(stringToNum<float>(tmp));
				tmp = "";
			}
		}
		else {
			if (tmp.length() > 0) {
				//data.push_back(tmp);
				buf.push_back(stringToNum<float>(tmp));
				tmp = "";
			}
		}
	}
	return buf;
}

double cosineMeasure(string file1, string file2)
{
	vector<float> buf1 = characteristicFile2vec(file1), buf2 = characteristicFile2vec(file2);
	return cosineMeasure(buf1, buf2);
}

double cosineMeasure(const vector<float>& buf1, const vector<float>& buf2)
{
	double mol = 0.0, deno1 = 0.0, deno2 = 0.0;
	for (auto i = 0; i < min((int)buf1.size(), (int)buf2.size()); i++)
	{
		//if(data1[i] != data2[i]) cout << i << " " << data1[i] << "    " << data2[i] << endl;
		mol += (double)buf1[i] * (double)buf2[i];
		deno1 += pow((double)buf1[i], 2);
		deno2 += pow((double)buf2[i], 2);
	}
	double val = mol / (sqrt(deno1) * sqrt(deno2));
	return val;
}

string getFileName(string& path)
{
	auto pos = path.find_last_of('/');
	if(pos < 0)
		pos = path.find_last_of('\\');
	auto pos1 = path.find_last_of('.');
	return path.substr(pos+1, pos1 - pos - 1);
}

const vector<pair<float, int>> accuracyVerification(string baseDir, string testDir, double step, vector<double> thresholds, string subKey)
{
	vector<string> filesbase, filestest;
	read_files_in_dir(baseDir.c_str(), filesbase);
	read_files_in_dir(testDir.c_str(), filestest);
	vector<pair<float, int>> result;
	vector<double> valtmp;
	map<string, vector<float>> basemap, testmap;

	LOGSTR(LDEBUG, "run characteristicFile2vec to init basemap and testmap");
	for (auto file : filesbase)
		basemap[file] = characteristicFile2vec(file);
	for (auto file : filestest)
		testmap[file] = characteristicFile2vec(file);

	LOGSTR(LDEBUG, "run cosineMeasure ");
	vector<thread> threads;
	mutex _mutex;
	for (auto i = 0; i < getCpus(); i++)
	{
		threads.push_back(thread([&]() {
			string filename = "", strtmp = "";
			vector<float> vals;
			double restmp = 0.0;
			while (true)
			{
				restmp = 0.0;
				strtmp = "";
				{
					lock_guard<mutex> lg(_mutex);
					auto it = testmap.begin();
					if(it==testmap.end())
						break;
					filename = it->first;
					vals = it->second;
					testmap.erase(it);
				}
				for (auto basenode : basemap)
				{
					auto res = cosineMeasure(vals, basenode.second);
					if (res > restmp)
					{
						restmp = res;
						strtmp = basenode.first;
					}
				}
				{
					lock_guard<mutex> lg(_mutex);
					valtmp.emplace_back(restmp);
				}
				filename = filename.substr(filename.rfind('/')+1);  
				strtmp = strtmp.substr(strtmp.rfind('/')+1);
				{
					int pos = (int)filename.find(".py.");
					if (pos > 0) filename.replace(pos, pos + 3, "");
					int pos1 = (int)strtmp.find(".py.");
					if (pos1 > 0) strtmp.replace(pos1, 3, "");
				}
				string filename1 = filename, strtmp1 = strtmp;
				if (subKey.length() > 0)
				{
					filename1 = filename.substr(0, filename.rfind(subKey.c_str()));
					strtmp1 = strtmp.substr(0, strtmp.rfind(subKey.c_str()));
				}
				if (filename1 != strtmp1)
					LOGSTR(LDEBUG, "file[" + filename + "] basefile[" + strtmp + "] res[" + to_string(restmp) + "]");
			}
		}));
	}
	for (auto i = 0; i < (int)threads.size(); i++)
		threads[i].join();
	sort(valtmp.begin(), valtmp.end());

	for (auto k = 0; k < (int)thresholds.size(); k++)
	{
		int count = 0, count1 = 0, size = (int)valtmp.size();
		double start = (int)(valtmp[0] / step) * step;
		for (auto i = 0; i < size; i++)
		{
			if (k == 0 && (valtmp[i] >= start + step || i + 1 == size))
			{
				cout << to_string(start) + "~" + to_string(start + step) + ":" + to_string(count) << endl;
				result.push_back(make_pair(start, count));
				start += step;
				count = 0;
			}
			if (valtmp[i] >= thresholds[k])
				count1++;
			++count;
		}
		cout << "------threshold:" + to_string(thresholds[k]) + "  rate:" + to_string(count1*100.0 / size) + "%" << endl;
	}
	return result;
}

string strFormat(string format, ...)
{
	char buf[1024] = { 0 };
	//sprintf(buf, format.c_str(), ....);
	return buf;
}

void* warpAndCropFace()
{
	const static cv::Size cropSize(96, 112);
	/*const static double REFERENCE_FACIAL_POINTS[5][2] = {
		{30.29459953, 51.69630051},
		{65.53179932, 51.50139999},
		{48.02519989, 71.73660278},
		{33.54930115, 92.3655014},
		{62.72990036, 92.20410156}
	};*/

	string file = "";
	Mat img = imread(file, 1);
	if (img.empty())
	{
		LOGSTR(LERROR, "imread " + file + " failed");
		return nullptr;
	}

	Mat tfm, outImg;
	warpAffine(img, outImg, tfm, cropSize);
	return nullptr;
}

float iou(float lbox[4], float rbox[4]) 
{
	float interBox[] = {
		(std::max)(lbox[0] - lbox[2] / 2.f , rbox[0] - rbox[2] / 2.f), //left
		(std::min)(lbox[0] + lbox[2] / 2.f , rbox[0] + rbox[2] / 2.f), //right
		(std::max)(lbox[1] - lbox[3] / 2.f , rbox[1] - rbox[3] / 2.f), //top
		(std::min)(lbox[1] + lbox[3] / 2.f , rbox[1] + rbox[3] / 2.f), //bottom
	};

	if (interBox[2] > interBox[3] || interBox[0] > interBox[1])
		return 0.0f;

	float interBoxS = (interBox[1] - interBox[0])*(interBox[3] - interBox[2]);
	return interBoxS / (lbox[2] * lbox[3] + rbox[2] * rbox[3] - interBoxS);
}

//bool cmp(const Yolo::Detection& a, const Yolo::Detection& b) 
//{
//	return a.conf > b.conf;
//}
//
//void nms(vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh /*= 0.5*/) 
//{
//	int det_size = sizeof(Yolo::Detection) / sizeof(float);
//	map<float, vector<Yolo::Detection>> m;
//	for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
//		if (output[1 + det_size * i + 4] <= conf_thresh) continue;
//		Yolo::Detection det;
//		memcpy(&det, &output[1 + det_size * i], det_size * sizeof(float));
//		if (m.count(det.class_id) == 0) m.emplace(det.class_id, vector<Yolo::Detection>());
//		m[det.class_id].push_back(det);
//	}
//	for (auto it = m.begin(); it != m.end(); it++) {
//		auto& dets = it->second;
//		std::sort(dets.begin(), dets.end(), cmp);
//		for (size_t m = 0; m < dets.size(); ++m) {
//			auto& item = dets[m];
//			res.push_back(item);
//			for (size_t n = m + 1; n < dets.size(); ++n) {
//				if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
//					dets.erase(dets.begin() + n);
//					--n;
//				}
//			}
//		}
//	}
//}

Rect getRect(int newWidth, int newHeight, float bbox[4], int width, int height) 
{
	int l, r, t, b;
	float r_w = width / (newWidth * 1.0);
	float r_h = height / (newHeight * 1.0);
	if (r_h > r_w) {
		l = bbox[0] - bbox[2] / 2.f;
		r = bbox[0] + bbox[2] / 2.f;
		t = bbox[1] - bbox[3] / 2.f - (height - r_w * newHeight) / 2;
		b = bbox[1] + bbox[3] / 2.f - (height - r_w * newHeight) / 2;
		l = l / r_w;
		r = r / r_w;
		t = t / r_w;
		b = b / r_w;
	}
	else {
		l = bbox[0] - bbox[2] / 2.f - (width - r_h * newWidth) / 2;
		r = bbox[0] + bbox[2] / 2.f - (width - r_h * newWidth) / 2;
		t = bbox[1] - bbox[3] / 2.f;
		b = bbox[1] + bbox[3] / 2.f;
		l = l / r_h;
		r = r / r_h;
		t = t / r_h;
		b = b / r_h;
	}
	return Rect(l, t, r - l, b - t);
}

Rect rectRecover(int oldW, int oldH, int newW, int newH, float bbox[4])
{
	float w, h, x, y;
	float r_w = newW / (oldW*1.0);
	float r_h = newH / (oldH*1.0);
	if (r_h > r_w) {
		w = newW;
		h = r_w * oldH;
		x = 0;
		y = (newH - h) / 2;
	}
	else {
		w = r_h * oldW;
		h = newH;
		x = (newW - w) / 2;
		y = 0;
	}
	Rect2f oldRc;
	oldRc.x = bbox[0] - bbox[2] / 2;
	oldRc.y = bbox[1] - bbox[3] / 2;
	oldRc.width = bbox[2];
	oldRc.height = bbox[3];

	oldRc.x -= x;
	oldRc.y -= y;

	auto ptx = oldRc.x + oldRc.width / 2, pty = oldRc.y + oldRc.height / 2;
	auto xrate = oldW / w, yrate = oldH / h;
	ptx *= xrate;
	pty *= yrate;
	oldRc.width *= xrate;
	oldRc.height *= yrate;
	oldRc.x = ptx - oldRc.width / 2;
	oldRc.y = pty - oldRc.height / 2;
	return oldRc;
}











int TimeStatistics::m_index = 1;
TimeStatistics::TimeStatistics(string prefix /*=""*/)
{
	m_prefix = prefix;
	m_tpindex = m_index++;
	m_tp = chrono::steady_clock::now();
	LOGSTR(LDEBUG, "-----" + m_prefix + "-enter---" + to_string(m_tpindex) + "-----------")
}
void TimeStatistics::destroy()
{
	if (m_lab) {
		auto dur = (chrono::duration<double, nano>(chrono::steady_clock::now() - m_tp)).count();
		LOGSTR(LDEBUG, "-----" + m_prefix + "-leave---" + to_string(m_tpindex) + "----------- duration:" + to_string(dur / 1e6) + "ms")
	}
	m_lab = false;
}





BatchImageProcess::BatchImageProcess(string path): m_path(path)
{
}

BatchImageProcess::~BatchImageProcess() 
{ 
	m_exit = true;
	LOGSTR(LDEBUG, "~BatchImageProcess");
}

bool BatchImageProcess::initialize(int batchSize, int width, int height, int channel, bool imgPadding /*= false*/)
{
	if (m_files.size() > 0) return true;
	m_batchSize = batchSize;
	m_slice_size = width * height * channel;
	m_data.reset(new float[batchSize * m_slice_size]);
	assert(m_data);
	read_files_in_dir(m_path.c_str(), m_files);
	if (m_files.size() > 0) {
		auto cpus = getCpus();
		auto cacheSize = max(3, 32 / batchSize) * batchSize;
		auto thrds = max(1, min(cpus, (cacheSize / cpus) < 4 ? 4 : (cacheSize / cpus)));
		LOGSTR(LDEBUG, "create thread num: " + to_string(thrds) + " cacheSize:" + to_string(cacheSize) + "  cpus:" + to_string(cpus));

		thread([=]() {
			int files_size = (int)m_files.size(), reset = m_reset;
			
			atomic<int> files_index(0);
			vector<thread> threads;
			for (auto i = 0; i < thrds; i++)
			{
				threads.emplace_back(thread([&](int thrd_index)
				{
					int count = 0;
					string name = "";
					while (files_index.load() < files_size && (m_files_limit <= 0 ? true : files_index.load() < m_files_limit) && !m_exit)
					{
						name = "";
						{
							if (reset != m_reset)
							{
								lock_guard<mutex> lg(m_mutex);
								if (reset != m_reset)
								{
									reset = m_reset;
									files_index.store(0);
									m_bufs.clear();
								}
							}
							if ((int)m_bufs.size() >= cacheSize)
								this_thread::yield();
							else {
								lock_guard<mutex> lg(m_mutex);
								auto _index = files_index.load();
								if (_index >= files_size || m_exit) break;
								name = m_files[_index];
								files_index++;
								count++;
							}
						}
						if (name.size() > 0)
						{
							BipResult bip;
							bip.file = name;
							imgConvert(bip.file, width, height, channel, bip.imgData, imgPadding, &bip.norSize);
							if (bip.imgData.size() > 0)
							{
								lock_guard<mutex> lg(m_mutex);
								m_bufs.emplace_back(bip);
							}
						}
					}
					LOGSTR(LDEBUG, "BatchImageProcess::initialize::thread" + to_string(thrd_index) + " exit, count: " + to_string(count) );
				}, i));
			}
			for (auto& thrd : threads) thrd.join();
			m_loopRet = true;
			LOGSTR(LDEBUG, "BatchImageProcess::initialize::thread_parent exit");
		}).detach();
		this_thread::sleep_for(chrono::milliseconds(50));
		return true;
	}
	return false;
}

int BatchImageProcess::getValidSize()
{
	lock_guard<mutex> lg(m_mutex);
	return (int)m_bufs.size();
}

bool BatchImageProcess::getData(float** data, int& dataSize, int& batchSize, vector<string>& files, vector<cv::Size>& norSizes)
{
	float* buf = m_data.get();
	memset(buf, 0, m_batchSize * m_slice_size);
	files.clear();
	norSizes.clear();
	*data = nullptr;
	dataSize = batchSize = 0;
	lock_guard<mutex> lg(m_mutex);
	int batch = min((int)m_bufs.size(), m_batchSize);
	if (batch < m_batchSize && !m_loopRet)
		return true;
	dataSize = batch * m_slice_size;
	batchSize = batch;
	for (auto i = 0; i < batch; i++)
	{
		auto node = &m_bufs.front();
		memcpy(buf + i * m_slice_size, node->imgData.data(), sizeof(float) * node->imgData.size());
		files.emplace_back(node->file);
		norSizes.emplace_back(node->norSize);
		m_bufs.pop_front();
	}
	*data = buf;
	return m_loopRet && batch < m_batchSize ? false : true;
}

static map<string, shared_ptr<BatchImageProcess>> BatchImageProcessors;
shared_ptr<BatchImageProcess> createBatchImageProcessor(string path)
{
	if (BatchImageProcessors.count(path) > 0)
		return BatchImageProcessors[path];
	shared_ptr<BatchImageProcess> pro(new BatchImageProcess(path));
	BatchImageProcessors[path] = pro;
	return pro;
}
void releaseBatchImageProcessor(string path)
{
	BatchImageProcessors.erase(path);
}
void releaseBatchImageProcessor(shared_ptr<BatchImageProcess> process)
{
	if (!process) return;
	for (auto& node : BatchImageProcessors)
	{
		if (node.second.get() == process.get())
		{
			BatchImageProcessors.erase(node.first);
			return;
		}
	}
}
