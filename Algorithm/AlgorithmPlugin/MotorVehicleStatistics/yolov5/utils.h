#ifndef TRTX_YOLOV5_UTILS_H_
#define TRTX_YOLOV5_UTILS_H_

#include "dirent.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <list>
#include <map>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>
#include <ratio>
#include <mutex>
#include <atomic>
#include "nlohmann/json.hpp"
#include <opencv2/opencv.hpp>
#include <sstream> 
#include "define.h"

static inline cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


/////add by sunth begin
using json = nlohmann::json;
using namespace std;
//using namespace cv;
#define FCL string(__FUNCTION__) + ":" + to_string(__LINE__)

const json& getCfg(const char* config);
cv::Mat imgResizeAndPadding(cv::Mat& img, int width, int height);
cv::Rect rectRecover(int oldW, int oldH, int newW, int newH, float bbox[4]);
bool imgConvert(string file, int width, int height, int channel, vector<float>& result, bool padding = false, cv::Size* norSize = nullptr);
bool imgConvert(cv::Mat img, int width, int height, int channel, vector<float>& result, bool padding = false, cv::Size* norSize = nullptr);
//bool listFiles(const string& path, vector<string>& files); use read_files_in_dir
void createTestJpg(int value, float inc, string path, int num);
string locateFile(string fileName, vector<string>& dirs);
string locateFolder(string folderName, vector<string>& dirs);
string buildPath(string fileName, string dir);
int getCpus();
//bool testFileExist(string file);
string getFileName(string& path);
vector<float> characteristicFile2vec(const string& file);
double cosineMeasure(string file1, string file2);
double cosineMeasure(const vector<float>& buf1, const vector<float>& buf2);
const vector<pair<float, int>> accuracyVerification(string baseDir, string testDir, double step, vector<double> thresholds, string subKey = "");
string strFormat(string format, ...);
void* warpAndCropFace();
float iou(float lbox[4], float rbox[4]);
//void nms(vector<Yolo::Detection>& res, float *output, float conf_thresh, float nms_thresh = 0.5);
cv::Rect getRect(int newWidth, int newHeight, float bbox[4], int width, int height);


class TimeStatistics
{
public:
	TimeStatistics(string prefix = "");
	~TimeStatistics() { destroy(); }
	void destroy();
private:
	chrono::steady_clock::time_point m_tp;
	int m_tpindex;
	string m_prefix;
	bool m_lab{ true };
	static int m_index;
};

//启动多线程，批量预处理文件夹中的图片
class BatchImageProcess
{
	struct BipResult
	{
		string file;
		cv::Size norSize;
		vector<float> imgData;
	};
public:
	BatchImageProcess(string path);
	virtual ~BatchImageProcess();

	bool initialize(int batchSize, int width, int height, int channel, bool imgPadding = false);
	int getValidSize();
	bool getData(float** data, int& dataSize, int& batchSize, vector<string>& files, vector<cv::Size>& norSizes);
	void resetIndex() { m_reset++; }
	void setLimit(int limit) { m_files_limit = limit; }

private:
	list<BipResult> m_bufs;
	vector<string> m_files;
	string m_path{ "" };
	int m_batchSize{ 0 };
	bool m_exit{ false };
	int m_reset{ 0 };
	UniquePtr1<float> m_data;
	int m_slice_size{ 0 };
	int m_files_limit{ 0 };
	std::mutex m_mutex;
	bool m_loopRet{ false };
};
shared_ptr<BatchImageProcess> createBatchImageProcessor(string path);
void releaseBatchImageProcessor(string path);
void releaseBatchImageProcessor(shared_ptr<BatchImageProcess> process);
//end
#endif  // TRTX_YOLOV5_UTILS_H_

