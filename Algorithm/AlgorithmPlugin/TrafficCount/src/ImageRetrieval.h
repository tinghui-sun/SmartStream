#pragma once
#include "Algorithm.h"
class ImageRetrieval : public AlgorithmIRInterface
{
public:
	ImageRetrieval(int gpuId);
	virtual ~ImageRetrieval();

public:
	virtual ALGOAbility getAbility() override;

	//�����ȶ� 1:1
	//srcFeature	�ȶ�����
	//dstFeature	���ȶ�����
	//similarity	���ƶ�ֵ��0-1�ĸ����ͣ�
	virtual ErrAlgorithm compare(const string& srcFeature, const string& dstFeature, float& similarity) override;

	//�����ȶ� N:M, N��M��������1
	//matchType ����ƥ�����ͣ�������ƥ����߶�����ƥ��
	//srcFeature Ԥ�ȶԵ������б�������һ���������߶������
	//dstFeature �׿������б�M������
	//threshold ��ֵ�������ֻ����������ڵ��ڸ�ֵ�Ľ��(0-1������)
	//limit �����б���ÿ�������ȶԺ󷵻ص���������������������limitС����ʵ����������
	//result ��ʶ����ߵļ�������ֵ
	virtual ErrAlgorithm compare(const IRMatchType& matchType, const std::list<IRFeatureInfo>& srcFeature, const std::list<IRFeatureInfo>&  dstFeature, const float& threshold, const uint32_t& limit, std::list<IRCompareResult>& result) override;

	//������ȡ����ȡimage������ֵ������������������ȶ�
	virtual ErrAlgorithm featureExtra(const ALGOImageInfo& imageInfo, IRFeatureInfo& fetureInfo) override;

	private:
	int mGpuId;
};

