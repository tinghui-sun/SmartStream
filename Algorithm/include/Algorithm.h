#pragma once

//IR (Image Retrieval)
//VA (Video Analysis)
#include "TypeDefAlgorithmError.h"
#include "AlgorithmLog.h"
#include <list>
#include <memory>
#include <string>
#include <map>
#include <vector>

using namespace std;

#define FEATURE_MAX_SIZE 512

//�㷨����
enum ALGOType
{
	ALGOTypeMultiTargetDetection = 0,	//��Ŀ�����㷨
	ALGOTypeFaceDetection,				//��������㷨			
	ALGOTypeFaceRecognition,			//����ʶ���㷨
	ALGOTypeBodyRecognition,			//����ʶ���㷨
	ALGOTypeBodyStatistics,				//����ͳ���㷨
	ALGOTypeMotorVehicleRecognition,	//������ʶ���㷨
	ALGOTypeMotorVehicleStatistics,		//����ͳ���㷨
	ALGOTypeNoMotorVehicleRecognition,	//�ǻ�����ʶ���㷨
	ALGOTypeSceneRecognition,			//����ʶ���㷨
	ALGOTypeThingRecognition,			//����ʶ���㷨
	ALGOTypePlateRecognition,			//����ʶ���㷨
	ALGOTypeCommonRecognition,			//ͨ��ʶ���㷨
	ALGOTypeVideoQualityDetection,		//��Ƶ�������
	ALGOTypePluginDemo,					//���ʾ��
	ALGOTypeLeaveDetection,				//��ڼ���㷨
	ALGOTypeMax							//
};

//ͼƬ��ʽ
enum ALGOImageFormat
{
	ALGOImageFormatBmp = 1,
	ALGOImageFormatGif = 2,
	ALGOImageFormatJpeg = 3,
	ALGOImageFormatJfif = 4,
	ALGOImageFormatKdc = 5,
	ALGOImageFormatPcd = 6,
	ALGOImageFormatPcx = 7,
	ALGOImageFormatPic = 8,
	ALGOImageFormatPix = 9,
	ALGOImageFormatPng = 10,
	ALGOImageFormatPsd = 11,
	ALGOImageFormatTapga = 12,
	ALGOImageFormatTiff = 13,
	ALGOImageFormatWmf = 14,
	ALGOImageFormatJp2 = 15,
	ALGOImageFormatCVMat = 16,
	ALGOImageFormatRGB = 17,
	ALGOImageFormatBRG = 18,
	ALGOImageFormatOther = 999
};

//���ͼƬ���ڴ��ַ����
enum ALGOBufferType
{
	ALGOBufferCPU = 1,
	ALGOBufferGPU = 2
};

//����
enum ALGOBodyType			//0110���ݳ̶�,
{
	ALGOBodyTypeVeryFat,	//0111����, Ƥ��֬���ر����, ��������,
	ALGOBodyTypeFat,		//0112����, Ƥ��֬������, ���ؽ���,
	ALGOBodyTypeNormal,		//0113�е�, �����ȳ�,
	ALGOBodyTypeThin,		//0114����, ���ⲻ����, Ƥ��֬���������ؽ���,
	ALGOBodyTypeVeryThin	//0115����, ��������, Ƥ��֬������, ���غ���, �׳ơ�Ƥ���ǡ��������°塱��,
};

//��ɫ
enum ALGOSkinColorType		//Ƥ������ɫ 2010
{
	ALGOSkinColorTypeWhite,	//�׷� 2011 Ƥ���ӽ���ɫ
	ALGOSkinColorTypeBlack,	//�ڷ� 2012 Ƥ�����ƺ�ɫ
	ALGOSkinColorTypeYellow,//�Ʒ� 2013 Ƥ�����ƻ�ɫ
	ALGOSkinColorTypeBrown,	//��ɫ 2014 Ƥ��������ɫ
};

//����
enum ALGOHairStyleType
{
	ALGOHairStyleType1,	//1	ƽͷ
	ALGOHairStyleType2, //2	�з�
	ALGOHairStyleType3, //3	ƫ��
	ALGOHairStyleType4, //4	��ͺ
	ALGOHairStyleType5, //5	��ͺ
	ALGOHairStyleType6, //6	ȫͺ
	ALGOHairStyleType7, //7	��
	ALGOHairStyleType8, //8	���˷�
	ALGOHairStyleType9, //9	�黨��
	ALGOHairStyleType10, //10 �̷�
	ALGOHairStyleType11, //11 ����
	ALGOHairStyleType12 //99 ����
};

//����
enum ALGOPostureType
{
	ALGOPostureType1,	//1	վ
	ALGOPostureType2,	//2	��
	ALGOPostureType3,	//3	��
	ALGOPostureType4,	//4	��
	ALGOPostureType5,	//5	��
	ALGOPostureType6,	//6	����
	ALGOPostureType7,	//7	����
	ALGOPostureType8,	//8	��Ծ
	ALGOPostureType9,	//9	�ʵ�
	ALGOPostureType10,	//10 ����
	ALGOPostureType11	//99 ����
};

//״̬
enum ALGOPersonStatusType
{
	ALGOPersonStatus1,	//1	 ���
	ALGOPersonStatus2,	//2	 ����
	ALGOPersonStatus3,	//3	 ����
	ALGOPersonStatus4	//99 ����
};


//����
enum ALGOFaceStyleType	//0310 ���������
{
	ALGOFaceStyleType1, //��Բ 0311 ������Բ�Σ�������Բ԰����Բ��
	ALGOFaceStyleType2, //Բǻ �������� 0312
	ALGOFaceStyleType3, //������ ���ʳ����� 0313
	ALGOFaceStyleType4, //�ճʷ��� ���� 0314
	ALGOFaceStyleType5, //������ 0315 ������խ�¿�������
	ALGOFaceStyleType6, //���ǹ� 0316 �����Ͽ���խ����������
	ALGOFaceStyleType7, //������ 0317 ���ĳ����֮�����Դ󣬳������Σ��׳�"����"��"�ݳ���"
	ALGOFaceStyleType8, //0318 ���Ϻ� �����ϡ���խ���м��������
	ALGOFaceStyleType9, //������ 0319 �����ṹ��ȱ�ݣ������������Բ��Գ�

	//0330 ��������
	ALGOFaceStyleType10, //ƽ��ǻ �沿������ƽ�� 0331
	ALGOFaceStyleType11, //������ 0332 ǰ��°�ͻ�����м��ڰ����׳�"������"
	ALGOFaceStyleType12	//͹���� 033 �����в�����ͻ��
};

//��������
enum ALGOFacialFeatureType
{

};

//��ò����
enum ALGOPhysicalFeatureType
{

};

//���
enum ALGOBodyFeatureType
{

};

//����
enum ALGOHabitualActionType
{
	ALGOHabitualActionType1,	//1	����
	ALGOHabitualActionType2,	//2	գ��
	ALGOHabitualActionType3,	//3	��ü
	ALGOHabitualActionType4,	//4	��ü
	ALGOHabitualActionType5,	//5	�ƾ�
	ALGOHabitualActionType6,	//6	ץͷ
	ALGOHabitualActionType7,	//7	�ڱ�
	ALGOHabitualActionType8,	//8	���°�
	ALGOHabitualActionType9,	//9	������
	ALGOHabitualActionType10,	//10��Ʋ��
	ALGOHabitualActionType11,	//11����
	ALGOHabitualActionType12,	//12��·ҡ��
	ALGOHabitualActionType13,	//13�����
	ALGOHabitualActionType14,	//14�ڰ���
	ALGOHabitualActionType15,	//15�漡�鴤
	ALGOHabitualActionType16,	//16˵������
	ALGOHabitualActionType17,	//17��ͷ
	ALGOHabitualActionType18,	//18�ֶ�
	ALGOHabitualActionType99	//99����
};

//��Ϊ
enum ALGOBehaviorType
{
	ALGOBehaviorType1,	//1	β��
	ALGOBehaviorType2,	//2	�ǻ�
	ALGOBehaviorType3,	//3	ȡ��
	ALGOBehaviorType4,	//4	���
	ALGOBehaviorType5,	//5	����
	ALGOBehaviorType6,	//6	���ֻ�
	ALGOBehaviorType99	//99����
};

//������
enum ALGOAppendageType
{
	ALGOAppendageTypeMobile,	//1	�ֻ�
	ALGOAppendageTypeUmbrella,	//2	ɡ
	ALGOAppendageTypeFacemask,	//3	����
	ALGOAppendageTypeWatch,	//4	�ֱ�
	ALGOAppendageTypeHelmet,	//5	ͷ��
	ALGOAppendageTypeGlasses,	//6	�۾�
	ALGOAppendageTypeHat,	//7	ñ��
	ALGOAppendageTypePackage,	//8	��
	ALGOAppendageTypeScarf,	//9	Χ��
	ALGOAppendageTypeOther	//99	����
};

//ñ����ʽ
enum ALGOHatStyleType
{
	ALGOHatStyleType1,	//1	ë��ñ
	ALGOHatStyleType2,  //2	����ñ
	ALGOHatStyleType3,  //3	����ñ
	ALGOHatStyleType4,  //4	ƽ��ñ
	ALGOHatStyleType5,  //5	���ñ
	ALGOHatStyleType6,  //6	��ͷñ
	ALGOHatStyleType7,  //7	Ѽ��ñ
	ALGOHatStyleType8,  //8	����ñ
	ALGOHatStyleTypeOther //99	����
};

//�۾���ʽ
enum ALGOGlassesStyleType
{
	ALGOGlassesStyleType1, //1	ȫ��
	ALGOGlassesStyleType2, //2	���
	ALGOGlassesStyleType3, //3	�޿�
	ALGOGlassesStyleType4, //4	ü�߿�
	ALGOGlassesStyleType5, //5	�๦�ܿ�
	ALGOGlassesStyleType6, //6	��ɫ��
	ALGOGlassesStyleType7, //7	̫����
	ALGOGlassesStyleType8, //8	�޾�Ƭ
	ALGOGlassesStyleType9, //9	͸��ɫ
	ALGOGlassesStyleTypeOther //99	����
};

//������ʽ
enum ALGOBagStyleType
{
	ALGOBagStyleType1,	//1	�����
	ALGOBagStyleType2,	//2	�����
	ALGOBagStyleType3,	//3	˫���
	ALGOBagStyleType4,	//4	Ǯ��
	ALGOBagStyleType5,	//5	���ð�
	ALGOBagStyleType6,	//6	����
	ALGOBagStyleType7,	//7	Կ�װ�
	ALGOBagStyleType8,	//8	����
	ALGOBagStyleType9,	//9	������
	ALGOBagStyleType10,	//10 ���а�
	ALGOBagStyleType11,	//11 ţ�а�
	ALGOBagStyleType12,	//12 б���
	ALGOBagStyleTypeOther	//99 ����
};

//������ʽ
enum ALGOCoatStyleType
{
	ALGOCoatStyleType1,	//1	��װ
	ALGOCoatStyleType2, //2	�����
	ALGOCoatStyleType3, //3	T��
	ALGOCoatStyleType4, //4	����
	ALGOCoatStyleType5, //5	����
	ALGOCoatStyleType6, //6	�п�
	ALGOCoatStyleType7, //7	Ƥ�п�
	ALGOCoatStyleType8, //8	����
	ALGOCoatStyleType9, //9	����
	ALGOCoatStyleType10, //10	ë��
	ALGOCoatStyleType11, //11	����
	ALGOCoatStyleType12, //12	���޷�
	ALGOCoatStyleType13, //13	�˶���
	ALGOCoatStyleType14, //14	������
	ALGOCoatStyleType15, //15	ţ�з�
	ALGOCoatStyleType16, //16	˯��
	ALGOCoatStyleType17, //17	����ȹ
	ALGOCoatStyleType18, //18	������
	ALGOCoatStyleTypeOther	//99	����
};

//���³���
enum ALGOCoatLengthType
{
	ALGOCoatLengthType1,	//1	����
	ALGOCoatLengthType2,	//2	����
	ALGOCoatLengthType3,	//3	����
	ALGOCoatLengthTypeOther		//3	����
};


//������ʽ
enum ALGOPantsStyleType
{
	ALGOPantsStyleType1,	//1	ţ�п�
	ALGOPantsStyleType2,	//2	����
	ALGOPantsStyleType3,	//3	��װ��
	ALGOPantsStyleType4,	//4	Ƥ��
	ALGOPantsStyleType5,	//5	ɳ̲��
	ALGOPantsStyleType6,	//6	�˶���
	ALGOPantsStyleType7,	//7	˯��
	ALGOPantsStyleType8,	//8	�޿���
	ALGOPantsStyleTypeOther	//99	����

};

//���ӳ���
enum ALGOPantsLengthType
{
	ALGOPantsLengthType1,	//1	����
	ALGOPantsLengthType2,	//2	�̿�
	ALGOPantsLengthTypeOther//99 ����
};


//Ь����ʽ
enum ALGOShoesStyleType
{
	ALGOShoesStyleType1,	//1	��Ь
	ALGOShoesStyleType2,	//2	ƤЬ
	ALGOShoesStyleType3,	//3	�˶�Ь
	ALGOShoesStyleType4,	//4	��Ь
	ALGOShoesStyleType5,	//5	��Ь
	ALGOShoesStyleType6,	//6	����Ь
	ALGOShoesStyleType7,	//7	��Ͳѥ
	ALGOShoesStyleType8,	//8	��Ͳѥ
	ALGOShoesStyleType9,	//9	��Ͳѥ
	ALGOShoesStyleType10,	//10��ɽѥ
	ALGOShoesStyleType11,	//11��װѥ
	ALGOShoesStyleType12,	//12��ѥ��
	ALGOShoesStyleTypeOther	//99	����
};

//��ɫ����
enum ALGOColorType
{
	ALGOColorTypeBlack, //1	1	��
	ALGOColorTypeWhite, //2	2	��
	ALGOColorTypegray, //3	3	��
	ALGOColorTypeRed, //4	4	��
	ALGOColorTypeBlue, //5	5	��
	ALGOColorTypeYellow, //6	6	��
	ALGOColorTypeOrange, //7	7	��
	ALGOColorTypeBrown, //8	8	��
	ALGOColorTypeGreen, //9	9	��
	ALGOColorTypePurple, //10	10	��
	ALGOColorTypeCyan, //11	11	��
	ALGOColorTypePink, //12	12	��
	ALGOColorTypeTransparent //13	13	͸��
};

//ʶ�𵽵���������
enum ALGOObjType
{
	ALGOObjTypeBody = 1,//����
	ALGOObjTypeFace = 2,//����
	ALGOObjTypeMotor = 3,//������
	ALGOObjTypeNoMotor = 4,//�ǻ�����
	ALGOObjTypeThing = 5,//��Ʒ
	ALGOObjTypeScene = 6,//����
	ALGOObjTypeDefault = 999//δ֪
};

//��������
enum ALGOBodyProperty
{
	ALGOBodyType = 0,		//1		����		BodyType		O
	ALGOBodySkinColor,		//2		��ɫ		SkinColorType	O
	ALGOBodyHairStyle,		//3		����		HairStyleType	O
	ALGOBodyHairColor,		//4		��ɫ		ColorType		O
	ALGOBodyGesture,		//5		��̬		PostureType		O
	ALGOPersonStatus,		//6		״̬		Status	PersonStatusType		O
	ALGOFaceStyle,			//7		����		ALGOBodyFaceStyle	FaceStyleType		O
	ALGOFacialFeature,		//8		��������	FacialFeature	FacialFeatureType		O
	ALGOPhysicalFeature,	//9		��ò����	PhysicalFeature	PhysicalFeatureType		O
	ALGOBodyFeature,		//10	�������	BodyFeature	BodyFeatureType		O
	ALGOHabitualAction,		//11	ϰ�߶���	HabitualMovement	HabitualActionType		O
	ALGOBehavior,			//12	��Ϊ		Behavior	BehaviorType		O
	ALGOBehaviorDescription,//13	��Ϊ����	BehaviorDescription	string	256	O
	ALGOAppendant,			//14	������		Appendant	AppendageType	128	O
	ALGOAppendantDesp,		//15	����������	AppendantDescription	string	256	O
	ALGOBodyUmbrellaColor,	//16	ɡ��ɫ		ColorType		O
	ALGOBodyRespiratorColor,//17	������ɫ	ColorType		O
	ALGOBodyCapStyle,		//18	ñ�ӿ�ʽ	HatStyleType		O
	ALGOBodyCapColor,		//19	ñ����ɫ	ColorType		O
	ALGOBodyGlassStyle,		//20	�۾���ʽ	GlassesStyleType		O
	ALGOBodyGlassColor,		//21	�۾���ɫ	ColorType		O
	ALGOBodyScarfColor,		//22	Χ����ɫ	ColorType		O
	ALGOBodyBagStyle,		//23	����ʽ		BagStyleType		O
	ALGOBodyBagColor,		//24	����ɫ		ColorType		O
	ALGOBodyCoatStyle,		//25	���¿�ʽ	CoatStyleType		O
	ALGOBodyCoatLength,		//26	���³���	CoatLength	CoatLengthType		O
	ALGOBodyCoatColor,		//27	������ɫ	ColorType		O
	ALGOBodyTrousersStyle,	//28	���ӿ�ʽ	PantsStyleType		O
	ALGOBodyTrousersColor,	//29	������ɫ	ColorType		O
	ALGOBodyTrousersLen,	//30	���ӳ���	TrousersLen	PantsLengthType		O
	ALGOBodyShoesStyle,		//31	Ь�ӿ�ʽ	ShoesStyleType		O
	ALGOBodyShoesColor		//32	Ь����ɫ	ColorType		O
};

//��������
enum ALGOFaceProperty
{
	ALGOFacePropertyPoint = 0,			//�����ؼ��� ������  ���� �������
	ALGOFacePropertySkinColor,			//1		��ɫ	SkinColor	SkinColorType
	ALGOFacePropertyHairStyle,			//2		����	HairStyle	HairStyleType
	ALGOFacePropertyHairColor,			//3		��ɫ	HairColor	ColorType
	ALGOFacePropertyFaceStyle,			//4		����	FaceStyle	FaceStyleType
	ALGOFacePropertyFacialFeature,		//5		��������	FacialFeature	FacialFeatureType
	ALGOFacePropertyPhysicalFeature,	//6		��ò����	PhysicalFeature	PhysicalFeatureType
	ALGOFacePropertyRespiratorColor,	//7		������ɫ	RespiratorColor	ColorType
	ALGOFacePropertyCapStyle,			//8		ñ�ӿ�ʽ	CapStyle	HatStyleType
	ALGOFacePropertyCapColor,			//9		ñ����ɫ	CapColor	ColorType
	ALGOFacePropertyGlassStyle,			//10	�۾���ʽ	GlassStyle	GlassesStyleType
	ALGOFacePropertyGlassColor			//11	�۾���ɫ	GlassColor	ColorType

};

//����������
enum ALGOMotorVehicleProperty
{
	ALGOMotorVehiclePropertyClass,	//��������	VehicleClass	VehicleClassType		O
	ALGOMotorVehiclePropertyBrand,	//����Ʒ��	VehicleBrand	VehicleBrandType		O	����ע������Ʒ��
	ALGOMotorVehiclePropertyModel,	//�����ͺ�	VehicleModel	VehicleModelType		O
	ALGOMotorVehiclePropertyStyles, //�������	VehicleStyles	string	16	O
	ALGOMotorVehiclePropertyLength,	//��������	VehicleLength	VehicleLengthType		int 5λ��������λΪ���ף�mm�� 
	ALGOMotorVehiclePropertyWidth,	//�������	VehicleWidth	VehicleWidthType		int 4λ��������λΪ���ף�mm��
	ALGOMotorVehiclePropertyHeight,	//�����߶�	VehicleHeight	VehicleHeightType		int 4λ��������λΪ���ף�mm��
	ALGOMotorVehiclePropertyColor,	//������ɫ	VehicleColor	ColorType		R
	ALGOMotorVehiclePropertyDepth	//��ɫ��ǳ	VehicleColorDepth	VehicleColorDepthType	 O
};

//GA / T16.4�������������ʹ���
const string ALGOVehicleClassTypeB10("B10"); //���Ͱ�ҳ�
const string ALGOVehicleClassTypeB11("B11"); // ������ͨ��ҳ�
const string ALGOVehicleClassTypeB12("B12"); // ������ʽ��ҳ�
const string ALGOVehicleClassTypeB13("B13"); // ���͹�ʽ��ҳ�
const string ALGOVehicleClassTypeB14("B14"); // ����ƽ���ҳ�
const string ALGOVehicleClassTypeB15("B15"); // ���ͼ�װ���ҳ�
const string ALGOVehicleClassTypeB16("B16"); // ������ж��ҳ�
const string ALGOVehicleClassTypeB17("B17"); // ��������ṹ��ҳ�
const string ALGOVehicleClassTypeB18("B18"); // ���Ͳ�դʽ��ҳ�
const string ALGOVehicleClassTypeB19("B19"); // �����þӰ�ҳ�
const string ALGOVehicleClassTypeB1A("B1A"); // ����ר����ҵ��ҳ�
const string ALGOVehicleClassTypeB1B("B1B"); // ���͵�ƽ���ҳ�
const string ALGOVehicleClassTypeB20("B20"); // ���Ͱ�ҳ�
const string ALGOVehicleClassTypeB21("B21"); // ������ͨ��ҳ�
const string ALGOVehicleClassTypeB22("B22"); // ������ʽ��ҳ�
const string ALGOVehicleClassTypeB23("B23"); // ���͹�ʽ��ҳ�
const string ALGOVehicleClassTypeB24("B24"); // ����ƽ���ҳ�
const string ALGOVehicleClassTypeB25("B25"); // ���ͼ�װ���ҳ�
const string ALGOVehicleClassTypeB26("B26"); // ������ж��ҳ�
const string ALGOVehicleClassTypeB27("B27"); // ��������ṹ��ҳ�
const string ALGOVehicleClassTypeB28("B28"); // ���Ͳ�դʽ��ҳ�
const string ALGOVehicleClassTypeB29("B29"); // �����þӰ�ҳ�
const string ALGOVehicleClassTypeB2A("B2A"); // ����ר����ҵ��ҳ�
const string ALGOVehicleClassTypeB2B("B2B"); // ���͵�ƽ���ҳ�
const string ALGOVehicleClassTypeB30("B30"); // ���Ͱ�ҳ�
const string ALGOVehicleClassTypeB31("B31"); // ������ͨ��ҳ�
const string ALGOVehicleClassTypeB32("B32"); // ������ʽ��ҳ�
const string ALGOVehicleClassTypeB33("B33"); // ���͹�ʽ��ҳ�
const string ALGOVehicleClassTypeB34("B34"); // ����ƽ���ҳ�
const string ALGOVehicleClassTypeB35("B35"); // ������ж��ҳ�
const string ALGOVehicleClassTypeB36("B36"); // ���Ͳ�դʽ��ҳ�
const string ALGOVehicleClassTypeB37("B37"); // �����þӰ�ҳ�
const string ALGOVehicleClassTypeB38("B38"); // ����ר����ҵ��ҳ�
const string ALGOVehicleClassTypeB39("B39"); // ���͵�ƽ���ҳ�
const string ALGOVehicleClassTypeD11("D11"); // �޹�糵
const string ALGOVehicleClassTypeD12("B12"); // �й�糵
const string ALGOVehicleClassTypeG10("G10"); // ����ȫ�ҳ�
const string ALGOVehicleClassTypeG11("G11"); // ������ͨȫ�ҳ�
const string ALGOVehicleClassTypeG12("G12"); // ������ʽȫ�ҳ�
const string ALGOVehicleClassTypeG13("G13"); // ���͹�ʽȫ�ҳ�
const string ALGOVehicleClassTypeG14("G14"); // ����ƽ��ȫ�ҳ�
const string ALGOVehicleClassTypeG15("G15"); // ���ͼ�װ��ȫ�ҳ�
const string ALGOVehicleClassTypeG16("G16"); // ������жȫ�ҳ�
const string ALGOVehicleClassTypeG17("G17"); // ���Ͳ�դʽȫ�ҳ�
const string ALGOVehicleClassTypeG18("G18"); // �����þ�ȫ�ҳ�
const string ALGOVehicleClassTypeG19("G19"); // ����ר����ҵȫ�ҳ�
const string ALGOVehicleClassTypeG20("G20"); // ����ȫ�ҳ�
const string ALGOVehicleClassTypeG21("G21"); // ������ͨȫ�ҳ�
const string ALGOVehicleClassTypeG22("G22"); // ������ʽȫ�ҳ�
const string ALGOVehicleClassTypeG23("G23"); // ���͹�ʽȫ�ҳ�
const string ALGOVehicleClassTypeG24("G24"); // ����ƽ��ȫ�ҳ�
const string ALGOVehicleClassTypeG25("G25"); // ���ͼ�װ��ȫ�ҳ�
const string ALGOVehicleClassTypeG26("G26"); // ������жȫ�ҳ�
const string ALGOVehicleClassTypeG27("G27"); // ���Ͳ�դʽȫ�ҳ�
const string ALGOVehicleClassTypeG28("G28"); // �����þ�ȫ�ҳ�
const string ALGOVehicleClassTypeG29("G29"); // ����ר����ҵȫ�ҳ�
const string ALGOVehicleClassTypeG30("G30"); // ����ȫ�ҳ�
const string ALGOVehicleClassTypeG31("G31"); // ������ͨȫ�ҳ�
const string ALGOVehicleClassTypeG32("G32"); // ������ʽȫ�ҳ�
const string ALGOVehicleClassTypeG33("G33"); // ���͹�ʽȫ�ҳ�
const string ALGOVehicleClassTypeG34("G34"); // ����ƽ��ȫ�ҳ�
const string ALGOVehicleClassTypeG35("G35"); // ������жȫ�ҳ�
const string ALGOVehicleClassTypeG36("G36"); // ���Ͳ�դʽȫ�ҳ�
const string ALGOVehicleClassTypeG37("G37"); // �����þ�ȫ�ҳ�
const string ALGOVehicleClassTypeG38("G38"); // ����ר����ҵȫ�ҳ�
const string ALGOVehicleClassTypeH10("H10"); // ���ͻ���
const string ALGOVehicleClassTypeH11("H11"); // ������ͨ����
const string ALGOVehicleClassTypeH12("H12"); // ������ʽ����
const string ALGOVehicleClassTypeH13("H13"); // ���ͷ�ջ���
const string ALGOVehicleClassTypeH14("H14"); // ���͹�ʽ����
const string ALGOVehicleClassTypeH15("H15"); // ����ƽ�����
const string ALGOVehicleClassTypeH16("H16"); // ���ͼ�װ�ᳵ
const string ALGOVehicleClassTypeH17("H17"); // ������ж����
const string ALGOVehicleClassTypeH18("H18"); // ��������ṹ����
const string ALGOVehicleClassTypeH19("H19"); // ���Ͳ�դʽ����
const string ALGOVehicleClassTypeH20("H20"); // ���ͻ���
const string ALGOVehicleClassTypeH21("H21"); // ������ͨ����
const string ALGOVehicleClassTypeH22("H22"); // ������ʽ����
const string ALGOVehicleClassTypeH23("H23"); // ���ͷ�ջ���
const string ALGOVehicleClassTypeH24("H24"); // ���͹�ʽ����
const string ALGOVehicleClassTypeH25("H25"); // ����ƽ�����
const string ALGOVehicleClassTypeH26("H26"); //���ͼ�װ�ᳵ
const string ALGOVehicleClassTypeH27("H27"); // ������ж����
const string ALGOVehicleClassTypeH28("H28"); // ��������ṹ����
const string ALGOVehicleClassTypeH29("H29"); // ���Ͳ�դʽ����
const string ALGOVehicleClassTypeH30("H30"); // ���ͻ���
const string ALGOVehicleClassTypeH31("H31"); // ������ͨ����
const string ALGOVehicleClassTypeH32("H32"); // ������ʽ����
const string ALGOVehicleClassTypeH33("H33"); // ���ͷ�ջ���
const string ALGOVehicleClassTypeH34("H34"); // ���͹�ʽ����
const string ALGOVehicleClassTypeH35("H35"); // ����ƽ�����
const string ALGOVehicleClassTypeH37("H37"); // ������ж����
const string ALGOVehicleClassTypeH38("H38"); // ��������ṹ����
const string ALGOVehicleClassTypeH39("H39"); // ���Ͳ�դʽ����
const string ALGOVehicleClassTypeH40("H40"); // ΢�ͻ���
const string ALGOVehicleClassTypeH41("H41"); // ΢����ͨ����
const string ALGOVehicleClassTypeH42("H42"); // ΢����ʽ����
const string ALGOVehicleClassTypeH43("H43"); // ΢�ͷ�ջ���
const string ALGOVehicleClassTypeH44("H44"); // ΢�͹�ʽ����
const string ALGOVehicleClassTypeH45("H45"); // ΢����ж����
const string ALGOVehicleClassTypeH46("H46"); // ΢������ṹ����
const string ALGOVehicleClassTypeH47("H47"); // ΢�Ͳ�դʽ����
const string ALGOVehicleClassTypeH50("H50"); // ���ٻ���
const string ALGOVehicleClassTypeH51("H51"); // ��ͨ���ٻ���
const string ALGOVehicleClassTypeH52("H52"); // ��ʽ���ٻ���
const string ALGOVehicleClassTypeH53("H53"); // ��ʽ���ٻ���
const string ALGOVehicleClassTypeH54("H54"); // ��ж���ٻ���
const string ALGOVehicleClassTypeH55("H55"); // ��դʽ���ٻ���
const string ALGOVehicleClassTypeJ11("J11"); // ��ʽװ�ػ�е
const string ALGOVehicleClassTypeJ12("J12"); // ��ʽ�ھ��е
const string ALGOVehicleClassTypeJ13("J13"); // ��ʽƽ�ػ�е
const string ALGOVehicleClassTypeK10("K10"); // ���Ϳͳ�
const string ALGOVehicleClassTypeK11("K11"); // ������ͨ�ͳ�
const string ALGOVehicleClassTypeK12("K12"); // ����˫��ͳ�
const string ALGOVehicleClassTypeK13("K13"); // �������̿ͳ�
const string ALGOVehicleClassTypeK14("K14"); // ���ͽ½ӿͳ�
const string ALGOVehicleClassTypeK15("K15"); // ����ԽҰ�ͳ�
const string ALGOVehicleClassTypeK16("K16"); // ���ͽγ�
const string ALGOVehicleClassTypeK17("K17"); // ����ר�ÿͳ�
const string ALGOVehicleClassTypeK20("K20"); // ���Ϳͳ�
const string ALGOVehicleClassTypeK21("K21"); // ������ͨ�ͳ�
const string ALGOVehicleClassTypeK22("K22"); // ����˫��ͳ�
const string ALGOVehicleClassTypeK23("K23"); // �������̿ͳ�
const string ALGOVehicleClassTypeK24("K24"); // ���ͽ½ӿͳ�
const string ALGOVehicleClassTypeK25("K25"); // ����ԽҰ�ͳ�
const string ALGOVehicleClassTypeK27("K27"); // ����ר�ÿͳ�
const string ALGOVehicleClassTypeK30("K30"); // С�Ϳͳ�
const string ALGOVehicleClassTypeK31("K31"); // С����ͨ�ͳ�
const string ALGOVehicleClassTypeK32("K32"); // С��ԽҰ�ͳ�
const string ALGOVehicleClassTypeK33("K33"); // С�ͽγ�
const string ALGOVehicleClassTypeK34("K34"); // С��ר�ÿͳ�
const string ALGOVehicleClassTypeK40("K40"); // ΢�Ϳͳ�
const string ALGOVehicleClassTypeK41("K41"); // ΢����ͨ�ͳ�
const string ALGOVehicleClassTypeK42("K42"); // ΢��ԽҰ�ͳ�
const string ALGOVehicleClassTypeK43("K43"); // ΢�ͽγ�
const string ALGOVehicleClassTypeM10("M10"); // ����Ħ�г�
const string ALGOVehicleClassTypeM11("M11"); // ��ͨ������Ħ�г�
const string ALGOVehicleClassTypeM12("M12"); // ���������Ħ�г�
const string ALGOVehicleClassTypeM13("M13"); // �������ؿ�Ħ�г�
const string ALGOVehicleClassTypeM14("M14"); // �������ػ�Ħ�г�
const string ALGOVehicleClassTypeM15("M15"); // ������Ħ�г�
const string ALGOVehicleClassTypeM20("M20"); // ����Ħ�г�
const string ALGOVehicleClassTypeM21("M21"); // ��ͨ����Ħ�г�
const string ALGOVehicleClassTypeM22("M22"); // ������Ħ�г�
const string ALGOVehicleClassTypeN11("N11"); // ��������
const string ALGOVehicleClassTypeQ10("Q10"); // ����ǣ����
const string ALGOVehicleClassTypeQ11("Q11"); // ���Ͱ��ǣ����
const string ALGOVehicleClassTypeQ12("Q12"); // ����ȫ��ǣ����
const string ALGOVehicleClassTypeQ20("Q20"); // ����ǣ����
const string ALGOVehicleClassTypeQ21("Q21"); // ���Ͱ��ǣ����
const string ALGOVehicleClassTypeQ22("Q22"); // ����ȫ��ǣ����
const string ALGOVehicleClassTypeQ30("Q30"); // ����ǣ����
const string ALGOVehicleClassTypeQ31("Q31"); // ���Ͱ��ǣ����
const string ALGOVehicleClassTypeQ32("Q32"); // ����ȫ��ǣ����
const string ALGOVehicleClassTypeT11("T11"); // ������ʽ������
const string ALGOVehicleClassTypeT20("T20"); // С��������
const string ALGOVehicleClassTypeT21("T21"); // С����ʽ������
const string ALGOVehicleClassTypeT22("T22"); // �ַ�������
const string ALGOVehicleClassTypeT23("T23"); // �ַ����������
const string ALGOVehicleClassTypeZ11("Z11"); // ����ר����ҵ��
const string ALGOVehicleClassTypeZ21("Z21"); // ����ר����ҵ��
const string ALGOVehicleClassTypeZ31("Z31"); // С��ר����ҵ��
const string ALGOVehicleClassTypeZ41("Z41"); // ΢��ר����ҵ��
const string ALGOVehicleClassTypeZ51("Z51"); // ����ר����ҵ��
const string ALGOVehicleClassTypeZ71("Z71"); // ����ר����ҵ��
const string ALGOVehicleClassTypeX99("B10"); // ����


enum ALGOVehicleBrandType
{
//0		����
//1		����
//2		���
//3		����
//4		����
//5		����
//6		����
//7		����
//8		�ղ�
//9		�µ�
//10	���Դ�
//11	ѩ����
//12	ѩ����
//13	�ִ�
//14	����
//15	����
//16	����
//17	����
//18	˹�´�
//19	����
//20	�л�
//21	�ֶ���
//22	�׿���˹
//23	������
//24	�����ۺ�
//25	����
//26	���ǵ�
//27	��ľ
//28	��
//29	����
//30	����
//31	����
//32	˹��³
//33	Ӣ��
//34	����
//35	����
//36	���壨��ʮ�壩
//37	����
//38	����
//39	����
//40	����
//41	����
//42	һ��
//43	��ά��
//44	����
//45	һ������
//46	�ʹ�
//47	��ŵ
//48	JMC
//49	MG����
//50	����
//51	��̩
//52	����
//53	���Ž���
//54	�Ϻ�����
//55	���ݽ���
//56	����
//57	��ͨ
//58	�й�����
//59	�����ؿ�
//60	������������
//61	Ծ������
//62	�ƺ�����
//65	��ʱ��
//66	��������
//67	Ӣ�����
//68	����ȫ��ӥ
//69	����
//70	·��
//71	�����Ա�
//73	ʱ������
//75	�����γ�
//76	�����ؿ�
//81	����
//82	����
//83	����
//86	��ͨ
//87	����
//88	��������
//89	��������
//90	½��
//92	����
//94	����
//95	ŷ��
//96	����
//97	����
//103	ک��
//104	����
//107	��������
//108	���ǽ�
//109	Ұ��
//110	����
//112	����˹��
//113	��������
//115	����
//117	�ݱ�
//119	�ƿ�ŷ��
//121	����
//122	����
//124	˫��
//128	��Դ
//136	����
//144	����
//155	��������
//167	�����ͳ�
//176	����
//191	����
//201	��ۿͳ�
//230	����
};

enum ALGOVehicleColorDepthType
{
	ALGOVehicleColorDepthType0,	//0	��
	ALGOVehicleColorDepthType1	//1	ǳ
};

//�ǻ���������
enum ALGONoMotorVehicleProperty
{
	//����Ʒ��	VehicleBrand	string	32	O	����ע������Ʒ��
	//��������	VehicleType		string	64	O	����ע�����Ŀ�ʽ�ͺ�����
	//��������	VehicleLength	VehicleLengthType		O
	//�������	VehicleWidth	VehicleWidthType		O
	//�����߶�	VehicleHeight	VehicleHeightType		O
	//������ɫ	VehicleColor	ColorType		R	��GA / T 543.5��DE00308
};

//��Ʒ����
enum ALGOTingProperty
{
	//��Ʒ����	Name	string	256	R / O	����ע��Ʒ����
	//��Ʒ��״	Shape	string	64	R / O	����ע��Ʒ��״����
	//��Ʒ��ɫ	Color	ColorType		R
	//��Ʒ��С	Size	string	64	O	����ע��Ʒ��С����
	//��Ʒ����	Material	string	256	O	����ע��Ʒ��������
};

//��������
enum ALGOSceneProperty
{

};

//����߽�
struct ALGOObjBounding
{
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
};

//��������ֵ����������
enum ALGOObjPropertyValueType
{
	ALGOObjPropertyTypeString = 0,
	ALGOObjPropertyTypeBool,
	ALGOObjPropertyTypeInt32,
	ALGOObjPropertyTypeUInt32,
	ALGOObjPropertyTypeInt64,
	ALGOObjPropertyTypeUInt64,
	ALGOObjPropertyTypeFloat,
	ALGOObjPropertyTypeDouble,
};

//ʶ�𵽵���������
struct ALGOObjProperty
{
	std::string propertyName;
	std::string propertyValue;
	ALGOObjPropertyValueType propertyValueType = ALGOObjPropertyTypeString;
};

//ʶ�𵽵�����
struct ALGOObjectParam
{
	int objectId = 0;		//�㷨�ڲ����ɵĶ���ID,�������ж��Ƿ�Ϊͬһ����
	int objType = 0;			//��������
	std::string objLabel;	//��ע���������
	float confidence = 0;	//��ʶ��0��1֮��ĸ�������Խ����1Խ����
	int roiId = 0;			//��ǰĿ������ROI����ID
	ALGOObjBounding boundingBox; //������ԭͼ�еı߽�
	std::list<ALGOObjProperty> propertyList; //ʶ�𵽵���������
};

//����ͼƬ����
struct ALGOImageInfo
{
	string imageId;					//����ͼƬ��ID,���㷨���������ɣ�ȫ��Ψһ
	ALGOImageFormat imageFormate = ALGOImageFormatJpeg; //ͼƬ��ʽ���μ�VCM_IMAGE_FORMAT_E
	int imageWidth = 0;				//����ͼƬ�Ŀ�
	int imageHeight = 0;			//����ͼƬ�ĸ�
	ALGOBufferType imageBufferType = ALGOBufferCPU; //���ͼƬ���ڴ��ַ����
	char* imageBuffer = nullptr;	//���㷨�����߷���,�㷨��������algorithmVAFinished�ٵ����ͷ�
	int imageBufferLen = 0;
	char* extend = nullptr;			//��չ�ֶ����㷨�����߷�����ͷ�
	int extendBufferLen = 0; 
};

struct ALGOVAResult
{	
	ErrAlgorithm code = ErrALGOSuccess;		//ͼƬ�������
	ALGOImageInfo imageInfo;				//����AlgorithmVAInterface::analyzeImageʱ����Ĵ�����ͼƬ��Ϣ
	map<ALGOObjType, list<ALGOObjectParam>> objParams;	//��������ͼƬ�е�������Ϣ,�����������ͷ���
	//list<ALGOObjectParam> objParams;		//��������ͼƬ�е�������Ϣ,�����������ͷ���
	int statisticsNum = 0;					//��Ա��������ͳ���㷨�õ���ͳ���������Ƕ�һϵ��ͼƬͳ�ƺ����ֵ����objParams����ȥ��
};

struct ROIPoint
{
	int x, y;
};

struct ROI
{
	int roiId = 0;
	string roiTitle;
	vector<ROIPoint> points;
};

//�ӿڲ�������
enum ALGOInterfaceParamValueType
{
	ALGOInterfaceParamType_JPEG_STRING,
	ALGOInterfaceParamType_OPENCV_MAT,
};

// ��ȡ�ӿڵ�����
struct ALGOAbility
{
	ALGOInterfaceParamValueType dataType;
};

enum IRMatchType
{
	IRMatchTypeLong, //�����������ȶ�,����Ŀ������������ȶ�
	IRMatchTypeShort, //�������ȶ�,�ȶԾ������������
};

struct IRFeatureInfo
{
	string imageId;	//������Ӧ��ͼƬID��ȫ��Ψһ

	float featureBuf[FEATURE_MAX_SIZE] = {0}; //����ֵ
	int featureLen = 0; //������������ڴ泤��
		
	float featureIndex[FEATURE_MAX_SIZE] = {0}; //���������׵�ַ
	int featureIndexLen = 0; //���������׵�ַ

	int quality; // ��ȡ������Ӧ��������������Χ0~100	
};

struct IRCompareResult
{
	string srcImageId;	//������Ӧ��ͼƬID��ȫ��Ψһ
	string dstImageId;	//������Ӧ��ͼƬID��ȫ��Ψһ
	float similarity;	//���ƶ�ֵ��0-1�ĸ����ͣ�
};

struct PluginParam
{
	string pluginPath;	//�������Ŀ¼����·����Ŀ¼�ṹ���㷨�Լ�����������pluginPath/conf ��Ų�������ļ�, pluginPath/model ���ģ�͡�
	shared_ptr<AlgoLoggerInterface> logger;
};

//Video Analysis�㷨��������ص�
class AlgorithmVAListener
{
public:
	AlgorithmVAListener() {};

	virtual ~AlgorithmVAListener() {};

	//���า�Ǹú�����ȡ�������
	//vaResult �㷨�����Ľ��
	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult) = 0;
};

//Video Analysis�㷨
class  AlgorithmVAInterface
{
public:
	AlgorithmVAInterface(){};

	virtual ~AlgorithmVAInterface() {};

	virtual ALGOAbility getAbility() = 0;

	//ע���첽�����Ļص�������
	void registerAVResListener(shared_ptr<AlgorithmVAListener> vaListener) { m_vaResListener = vaListener; }

	//�첽��������
	virtual ErrAlgorithm analyzeImageASync(const std::list<ALGOImageInfo>& imageList) = 0;

	//ͬ����������
	virtual ErrAlgorithm analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResult) = 0;

	
	
protected:
	//Video Analysis����������Ļص�����
	weak_ptr<AlgorithmVAListener> m_vaResListener;
};

//Image Retrieval�㷨
class  AlgorithmIRInterface
{
public:
	AlgorithmIRInterface() {};

	virtual ~AlgorithmIRInterface() {};
	virtual ALGOAbility getAbility() = 0;

	//�����ȶ� 1:1
	//srcFeature	�ȶ�����
	//dstFeature	���ȶ�����
	//similarity	���ƶ�ֵ��0-1�ĸ����ͣ�
	virtual ErrAlgorithm compare(const string& srcFeature, const string& dstFeature, float& similarity) = 0;

	//�����ȶ� N:M, N��M��������1
	//matchType ����ƥ�����ͣ�������ƥ����߶�����ƥ��
	//srcFeature Ԥ�ȶԵ������б�������һ���������߶������
	//dstFeature �׿������б�M������
	//threshold ��ֵ�������ֻ����������ڵ��ڸ�ֵ�Ľ��(0-1������)
	//limit �����б���ÿ�������ȶԺ󷵻ص���������������������limitС����ʵ����������
	//result ��ʶ����ߵļ�������ֵ
	virtual ErrAlgorithm compare(const IRMatchType& matchType, const std::list<IRFeatureInfo>& srcFeature, const std::list<IRFeatureInfo>&  dstFeature, const float& threshold, const uint32_t& limit, std::list<IRCompareResult>& result) = 0;

	//������ȡ����ȡimage������ֵ������������������ȶ�
	virtual ErrAlgorithm featureExtra(const ALGOImageInfo& imageInfo, IRFeatureInfo& fetureInfo) = 0;
};


class  AlgorithmPluginInterface
{
public:
	AlgorithmPluginInterface() {};

	virtual ~AlgorithmPluginInterface() {};

	//��ʼ���㷨���
	virtual ErrAlgorithm pluginInitialize(const PluginParam& pluginParam, int gpuId) = 0;

	//�ͷ��㷨���
	virtual ErrAlgorithm pluginRelease() = 0;

	//����������Video Analysis�㷨
	//gpuId < 0 :ʹ��CPU
	//gpuId > 0 :ʹ��ָ��GPU
	virtual shared_ptr<AlgorithmVAInterface> createVAAlgorithm(int gpuId) = 0;
	virtual void destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo) = 0;

	//����������Image Retrieval�㷨
	//gpuId < 0 :ʹ��CPU
	//gpuId > 0 :ʹ��ָ��GPU
	virtual shared_ptr<AlgorithmIRInterface> createIRAlgorithm(int gpuId) = 0;
	virtual void destoryIRAlgorithm(shared_ptr<AlgorithmIRInterface> algo) = 0;
	
};