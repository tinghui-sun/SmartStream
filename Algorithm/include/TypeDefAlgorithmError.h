#pragma once

enum ErrAlgorithm
{
	//�ɹ�
	ErrALGOSuccess = 0,
	
	//�ֱ��ʲ�֧��
	ErrALGOResolutionUnSupport = -101,

	//�Ƿ�����
	ErrALGOParamInvalid = -102,

	//���ܲ�֧��
	ErrALGOUnSupport = -103,

	//�����ʼ��ʧ��
	ErrALGOInitFailed = -104,

	//�㷨ִ��ʧ��
	ErrALGORunFailed = -105,

	// �����cv::MatΪ��
	ErrALGOMatEmpty = -110,

	// cv::MatԤ����ת��ʧ��
	ErrALGOMatConvert = -111,

	// ����ڲ���ֵ
	ErrALGOOBJNULLPTR = -112,

	// �����ļ�����ȷ
	ErrALGOConfigInvalid = -113,

	//λ�ô���
	ErrALGOUnknow = -199,
};
