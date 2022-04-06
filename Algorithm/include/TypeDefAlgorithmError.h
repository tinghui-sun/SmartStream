#pragma once

enum ErrAlgorithm
{
	//成功
	ErrALGOSuccess = 0,
	
	//分辨率不支持
	ErrALGOResolutionUnSupport = -101,

	//非法参数
	ErrALGOParamInvalid = -102,

	//功能不支持
	ErrALGOUnSupport = -103,

	//插件初始化失败
	ErrALGOInitFailed = -104,

	//算法执行失败
	ErrALGORunFailed = -105,

	// 构造的cv::Mat为空
	ErrALGOMatEmpty = -110,

	// cv::Mat预处理转换失败
	ErrALGOMatConvert = -111,

	// 插件内部空值
	ErrALGOOBJNULLPTR = -112,

	// 配置文件不正确
	ErrALGOConfigInvalid = -113,

	//位置错误
	ErrALGOUnknow = -199,
};
