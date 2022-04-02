#pragma once
/******************************************
* 错误码头文件		错误码格式： 前两位模块编号-后三位错误码编号。最多支持99个模块，每个模块最多999个错误码
* GBErrCommon.h		通用错误码 -10000:-10999
* GBErrClientSDK.h	客户端SDK错误 -11000:-11999
* GBErrServerSDK.h	服务端SDK错误 -12000:-12999
* GBErrCCS.h		CCS错误 -13000:-13999
* GBErrMDS.h		MDS错误码 -14000:-14999
* GBErrNRS.h		NRS错误码 -15000:-15999
* GBErrEVS.h		EVS错误码 -16000:-16999
* GBErrPAG.h		PAG错误码 -17000:-17999
* GBErrPlayer.h		PLayer错误码 -18000:-18999
* GBErrSip.h		Sip错误码 -19000:-19999
* GBErrMSE.h		MSE错误码 -20000:-20999
* GBErrDevSDK.h		MSE错误码 -21000:-21999
*******************************************/

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

	//位置错误
	ErrALGOUnknow = -199,
};
