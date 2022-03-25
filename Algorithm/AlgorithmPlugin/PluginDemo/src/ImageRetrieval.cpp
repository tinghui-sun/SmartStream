#include "ImageRetrieval.h"
#include "GlobalParm.h"

ImageRetrieval::ImageRetrieval()
{
}


ImageRetrieval::~ImageRetrieval()
{
}

ALGOAbility ImageRetrieval::getAbility()
{
	ALGOAbility ability;
	ability.dataType = ALGOInterfaceParamType_JPEG_STRING;
	return ability;
}

//特征比对 1:1
//srcFeature	比对特征
//dstFeature	被比对特征
//similarity	相似度值（0-1的浮点型）
ErrAlgorithm ImageRetrieval::compare(const string& srcFeature, const string& dstFeature, float& similarity)
{
	return ErrALGOSuccess;
}

//特征比对 N:M, N和M都可以是1
//matchType 特征匹配类型，长特征匹配或者短特征匹配
//srcFeature 预比对的特征列表，可以是一个特征或者多个特征
//dstFeature 底库特征列表，M个特征
//threshold 阈值，结果中只输出分数大于等于该值的结果(0-1浮点型)
//limit 特征列表中每个特征比对后返回的最大结果数据量，若结果比limit小，按实际数量即可
//result 相识度最高的几个特征值
ErrAlgorithm ImageRetrieval::compare(const IRMatchType& matchType, const std::list<IRFeatureInfo>& srcFeature, const std::list<IRFeatureInfo>&  dstFeature, const float& threshold, const uint32_t& limit, std::list<IRCompareResult>& result)
{
	return ErrALGOSuccess;
}

//特征提取，提取image的特征值，后面可用来做特征比对
ErrAlgorithm ImageRetrieval::featureExtra(const ALGOImageInfo& imageInfo, IRFeatureInfo& fetureInfo)
{
	return ErrALGOSuccess;
}