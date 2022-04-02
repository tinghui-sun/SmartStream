#pragma once
/******************************************
* ������ͷ�ļ�		�������ʽ�� ǰ��λģ����-����λ�������š����֧��99��ģ�飬ÿ��ģ�����999��������
* GBErrCommon.h		ͨ�ô����� -10000:-10999
* GBErrClientSDK.h	�ͻ���SDK���� -11000:-11999
* GBErrServerSDK.h	�����SDK���� -12000:-12999
* GBErrCCS.h		CCS���� -13000:-13999
* GBErrMDS.h		MDS������ -14000:-14999
* GBErrNRS.h		NRS������ -15000:-15999
* GBErrEVS.h		EVS������ -16000:-16999
* GBErrPAG.h		PAG������ -17000:-17999
* GBErrPlayer.h		PLayer������ -18000:-18999
* GBErrSip.h		Sip������ -19000:-19999
* GBErrMSE.h		MSE������ -20000:-20999
* GBErrDevSDK.h		MSE������ -21000:-21999
*******************************************/

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

	//λ�ô���
	ErrALGOUnknow = -199,
};
