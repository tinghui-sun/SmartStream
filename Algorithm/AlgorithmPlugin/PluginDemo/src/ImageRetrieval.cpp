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

//�����ȶ� 1:1
//srcFeature	�ȶ�����
//dstFeature	���ȶ�����
//similarity	���ƶ�ֵ��0-1�ĸ����ͣ�
ErrAlgorithm ImageRetrieval::compare(const string& srcFeature, const string& dstFeature, float& similarity)
{
	return ErrALGOSuccess;
}

//�����ȶ� N:M, N��M��������1
//matchType ����ƥ�����ͣ�������ƥ����߶�����ƥ��
//srcFeature Ԥ�ȶԵ������б�������һ���������߶������
//dstFeature �׿������б�M������
//threshold ��ֵ�������ֻ����������ڵ��ڸ�ֵ�Ľ��(0-1������)
//limit �����б���ÿ�������ȶԺ󷵻ص���������������������limitС����ʵ����������
//result ��ʶ����ߵļ�������ֵ
ErrAlgorithm ImageRetrieval::compare(const IRMatchType& matchType, const std::list<IRFeatureInfo>& srcFeature, const std::list<IRFeatureInfo>&  dstFeature, const float& threshold, const uint32_t& limit, std::list<IRCompareResult>& result)
{
	return ErrALGOSuccess;
}

//������ȡ����ȡimage������ֵ������������������ȶ�
ErrAlgorithm ImageRetrieval::featureExtra(const ALGOImageInfo& imageInfo, IRFeatureInfo& fetureInfo)
{
	return ErrALGOSuccess;
}