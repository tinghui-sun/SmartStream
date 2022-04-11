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

//算法类型
enum ALGOType
{
	ALGOTypeMultiTargetDetection = 0,	//多目标检测算法
	ALGOTypeFaceDetection,				//人脸检测算法			
	ALGOTypeFaceRecognition,			//人脸识别算法
	ALGOTypeBodyRecognition,			//人体识别算法
	ALGOTypeBodyStatistics,				//人体统计算法
	ALGOTypeMotorVehicleRecognition,	//机动车识别算法
	ALGOTypeMotorVehicleStatistics,		//车辆统计算法
	ALGOTypeNoMotorVehicleRecognition,	//非机动车识别算法
	ALGOTypeSceneRecognition,			//场景识别算法
	ALGOTypeThingRecognition,			//物体识别算法
	ALGOTypePlateRecognition,			//车牌识别算法
	ALGOTypeCommonRecognition,			//通用识别算法
	ALGOTypeVideoQualityDetection,		//视频质量检测
	ALGOTypePluginDemo,					//插件示例
	ALGOTypeLeaveDetection,				//离岗检测算法
	ALGOTypeMax							//
};

//图片格式
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

//存放图片的内存地址类型
enum ALGOBufferType
{
	ALGOBufferCPU = 1,
	ALGOBufferGPU = 2
};

//体型
enum ALGOBodyType			//0110胖瘦程度,
{
	ALGOBodyTypeVeryFat,	//0111特胖, 皮下脂肪特别丰满, 体重特重,
	ALGOBodyTypeFat,		//0112较胖, 皮下脂肪丰满, 体重较重,
	ALGOBodyTypeNormal,		//0113中等, 体型匀称,
	ALGOBodyTypeThin,		//0114较瘦, 肌肉不发达, 皮下脂肪不多体重较轻,
	ALGOBodyTypeVeryThin	//0115特瘦, 发育不良, 皮下脂肪很少, 体重很轻, 俗称“皮包骨”、“搓衣板”等,
};

//肤色
enum ALGOSkinColorType		//皮肤的颜色 2010
{
	ALGOSkinColorTypeWhite,	//白肤 2011 皮肤接近白色
	ALGOSkinColorTypeBlack,	//黑肤 2012 皮肤近似黑色
	ALGOSkinColorTypeYellow,//黄肤 2013 皮肤近似黄色
	ALGOSkinColorTypeBrown,	//棕色 2014 皮肤近似棕色
};

//发型
enum ALGOHairStyleType
{
	ALGOHairStyleType1,	//1	平头
	ALGOHairStyleType2, //2	中分
	ALGOHairStyleType3, //3	偏分
	ALGOHairStyleType4, //4	额秃
	ALGOHairStyleType5, //5	项秃
	ALGOHairStyleType6, //6	全秃
	ALGOHairStyleType7, //7	卷发
	ALGOHairStyleType8, //8	波浪发
	ALGOHairStyleType9, //9	麻花辫
	ALGOHairStyleType10, //10 盘发
	ALGOHairStyleType11, //11 披肩
	ALGOHairStyleType12 //99 其他
};

//姿势
enum ALGOPostureType
{
	ALGOPostureType1,	//1	站
	ALGOPostureType2,	//2	蹲
	ALGOPostureType3,	//3	卧
	ALGOPostureType4,	//4	躺
	ALGOPostureType5,	//5	坐
	ALGOPostureType6,	//6	行走
	ALGOPostureType7,	//7	奔跑
	ALGOPostureType8,	//8	跳跃
	ALGOPostureType9,	//9	攀登
	ALGOPostureType10,	//10 匍匐
	ALGOPostureType11	//99 其他
};

//状态
enum ALGOPersonStatusType
{
	ALGOPersonStatus1,	//1	 醉酒
	ALGOPersonStatus2,	//2	 亢奋
	ALGOPersonStatus3,	//3	 正常
	ALGOPersonStatus4	//99 其他
};


//脸型
enum ALGOFaceStyleType	//0310 正面观脸型
{
	ALGOFaceStyleType1, //满圆 0311 脸呈椭圆形，包括长圆园、卵圆形
	ALGOFaceStyleType2, //圆腔 脸呈阔形 0312
	ALGOFaceStyleType3, //长方脸 脸呈长方形 0313
	ALGOFaceStyleType4, //险呈方形 方险 0314
	ALGOFaceStyleType5, //倒大脸 0315 脸部上窄下宽，呈梯形
	ALGOFaceStyleType6, //三角股 0316 脸部上宽下窄，皇三角形
	ALGOFaceStyleType7, //狭长脸 0317 脸的长与宽之比明显大，呈狭长形，俗称"马脸"或"瘦长险"
	ALGOFaceStyleType8, //0318 喘诚喝 脸部上、下窄，中间宽，呈菱形
	ALGOFaceStyleType9, //畸形脸 0319 睑部结构有缺陷，如左、右脸明显不对称

	//0330 侧面脸型
	ALGOFaceStyleType10, //平形腔 面部轮廓呈平面 0331
	ALGOFaceStyleType11, //凹形脸 0332 前额、下巴突出，中间内凹，俗称"月牙脸"
	ALGOFaceStyleType12	//凸形验 033 脸的中部明显突出
};

//脸部特征
enum ALGOFacialFeatureType
{

};

//体貌特征
enum ALGOPhysicalFeatureType
{

};

//体表
enum ALGOBodyFeatureType
{

};

//动作
enum ALGOHabitualActionType
{
	ALGOHabitualActionType1,	//1	翻眼
	ALGOHabitualActionType2,	//2	眨眼
	ALGOHabitualActionType3,	//3	皱眉
	ALGOHabitualActionType4,	//4	挑眉
	ALGOHabitualActionType5,	//5	推镜
	ALGOHabitualActionType6,	//6	抓头
	ALGOHabitualActionType7,	//7	挖鼻
	ALGOHabitualActionType8,	//8	摸下巴
	ALGOHabitualActionType9,	//9	打手势
	ALGOHabitualActionType10,	//10左撇子
	ALGOHabitualActionType11,	//11缩颈
	ALGOHabitualActionType12,	//12走路摇摆
	ALGOHabitualActionType13,	//13外八字
	ALGOHabitualActionType14,	//14内八字
	ALGOHabitualActionType15,	//15面肌抽搐
	ALGOHabitualActionType16,	//16说话歪嘴
	ALGOHabitualActionType17,	//17摆头
	ALGOHabitualActionType18,	//18手抖
	ALGOHabitualActionType99	//99其他
};

//行为
enum ALGOBehaviorType
{
	ALGOBehaviorType1,	//1	尾随
	ALGOBehaviorType2,	//2	徘徊
	ALGOBehaviorType3,	//3	取款
	ALGOBehaviorType4,	//4	打架
	ALGOBehaviorType5,	//5	开车
	ALGOBehaviorType6,	//6	打手机
	ALGOBehaviorType99	//99其他
};

//附属物
enum ALGOAppendageType
{
	ALGOAppendageTypeMobile,	//1	手机
	ALGOAppendageTypeUmbrella,	//2	伞
	ALGOAppendageTypeFacemask,	//3	口罩
	ALGOAppendageTypeWatch,	//4	手表
	ALGOAppendageTypeHelmet,	//5	头盔
	ALGOAppendageTypeGlasses,	//6	眼镜
	ALGOAppendageTypeHat,	//7	帽子
	ALGOAppendageTypePackage,	//8	包
	ALGOAppendageTypeScarf,	//9	围巾
	ALGOAppendageTypeOther	//99	其他
};

//帽子样式
enum ALGOHatStyleType
{
	ALGOHatStyleType1,	//1	毛线帽
	ALGOHatStyleType2,  //2	贝雷帽
	ALGOHatStyleType3,  //3	棒球帽
	ALGOHatStyleType4,  //4	平项帽
	ALGOHatStyleType5,  //5	渔夫帽
	ALGOHatStyleType6,  //6	套头帽
	ALGOHatStyleType7,  //7	鸭舌帽
	ALGOHatStyleType8,  //8	大檐帽
	ALGOHatStyleTypeOther //99	其他
};

//眼睛样式
enum ALGOGlassesStyleType
{
	ALGOGlassesStyleType1, //1	全框
	ALGOGlassesStyleType2, //2	半框
	ALGOGlassesStyleType3, //3	无框
	ALGOGlassesStyleType4, //4	眉线框
	ALGOGlassesStyleType5, //5	多功能框
	ALGOGlassesStyleType6, //6	变色镜
	ALGOGlassesStyleType7, //7	太阳镜
	ALGOGlassesStyleType8, //8	无镜片
	ALGOGlassesStyleType9, //9	透明色
	ALGOGlassesStyleTypeOther //99	其他
};

//背包样式
enum ALGOBagStyleType
{
	ALGOBagStyleType1,	//1	单肩包
	ALGOBagStyleType2,	//2	手提包
	ALGOBagStyleType3,	//3	双肩包
	ALGOBagStyleType4,	//4	钱包
	ALGOBagStyleType5,	//5	手拿包
	ALGOBagStyleType6,	//6	腰包
	ALGOBagStyleType7,	//7	钥匙包
	ALGOBagStyleType8,	//8	卡包
	ALGOBagStyleType9,	//9	手拉箱
	ALGOBagStyleType10,	//10 旅行包
	ALGOBagStyleType11,	//11 牛仔包
	ALGOBagStyleType12,	//12 斜挎包
	ALGOBagStyleTypeOther	//99 其他
};

//上衣样式
enum ALGOCoatStyleType
{
	ALGOCoatStyleType1,	//1	西装
	ALGOCoatStyleType2, //2	民族服
	ALGOCoatStyleType3, //3	T恤
	ALGOCoatStyleType4, //4	衬衫
	ALGOCoatStyleType5, //5	卫衣
	ALGOCoatStyleType6, //6	夹克
	ALGOCoatStyleType7, //7	皮夹克
	ALGOCoatStyleType8, //8	大衣
	ALGOCoatStyleType9, //9	风衣
	ALGOCoatStyleType10, //10	毛衣
	ALGOCoatStyleType11, //11	棉衣
	ALGOCoatStyleType12, //12	羽绒服
	ALGOCoatStyleType13, //13	运动服
	ALGOCoatStyleType14, //14	工作服
	ALGOCoatStyleType15, //15	牛仔服
	ALGOCoatStyleType16, //16	睡衣
	ALGOCoatStyleType17, //17	连衣裙
	ALGOCoatStyleType18, //18	无上衣
	ALGOCoatStyleTypeOther	//99	其他
};

//上衣长度
enum ALGOCoatLengthType
{
	ALGOCoatLengthType1,	//1	长袖
	ALGOCoatLengthType2,	//2	短袖
	ALGOCoatLengthType3,	//3	无袖
	ALGOCoatLengthTypeOther		//3	无袖
};


//裤子样式
enum ALGOPantsStyleType
{
	ALGOPantsStyleType1,	//1	牛仔裤
	ALGOPantsStyleType2,	//2	西裤
	ALGOPantsStyleType3,	//3	工装裤
	ALGOPantsStyleType4,	//4	皮裤
	ALGOPantsStyleType5,	//5	沙滩裤
	ALGOPantsStyleType6,	//6	运动裤
	ALGOPantsStyleType7,	//7	睡裤
	ALGOPantsStyleType8,	//8	无裤子
	ALGOPantsStyleTypeOther	//99	其他

};

//裤子长度
enum ALGOPantsLengthType
{
	ALGOPantsLengthType1,	//1	长裤
	ALGOPantsLengthType2,	//2	短裤
	ALGOPantsLengthTypeOther//99 其他
};


//鞋子样式
enum ALGOShoesStyleType
{
	ALGOShoesStyleType1,	//1	板鞋
	ALGOShoesStyleType2,	//2	皮鞋
	ALGOShoesStyleType3,	//3	运动鞋
	ALGOShoesStyleType4,	//4	拖鞋
	ALGOShoesStyleType5,	//5	凉鞋
	ALGOShoesStyleType6,	//6	休闲鞋
	ALGOShoesStyleType7,	//7	高筒靴
	ALGOShoesStyleType8,	//8	中筒靴
	ALGOShoesStyleType9,	//9	低筒靴
	ALGOShoesStyleType10,	//10登山靴
	ALGOShoesStyleType11,	//11军装靴
	ALGOShoesStyleType12,	//12无靴子
	ALGOShoesStyleTypeOther	//99	其他
};

//颜色种类
enum ALGOColorType
{
	ALGOColorTypeBlack, //1	1	黑
	ALGOColorTypeWhite, //2	2	白
	ALGOColorTypegray, //3	3	灰
	ALGOColorTypeRed, //4	4	红
	ALGOColorTypeBlue, //5	5	蓝
	ALGOColorTypeYellow, //6	6	黄
	ALGOColorTypeOrange, //7	7	橙
	ALGOColorTypeBrown, //8	8	棕
	ALGOColorTypeGreen, //9	9	绿
	ALGOColorTypePurple, //10	10	紫
	ALGOColorTypeCyan, //11	11	青
	ALGOColorTypePink, //12	12	粉
	ALGOColorTypeTransparent //13	13	透明
};

//识别到的物体类型
enum ALGOObjType
{
	ALGOObjTypeBody = 1,//人体
	ALGOObjTypeFace = 2,//人脸
	ALGOObjTypeMotor = 3,//机动车
	ALGOObjTypeNoMotor = 4,//非机动车
	ALGOObjTypeThing = 5,//物品
	ALGOObjTypeScene = 6,//场景
	ALGOObjTypeDefault = 999//未知
};

//人体属性
enum ALGOBodyProperty
{
	ALGOBodyType = 0,		//1		体型		BodyType		O
	ALGOBodySkinColor,		//2		肤色		SkinColorType	O
	ALGOBodyHairStyle,		//3		发型		HairStyleType	O
	ALGOBodyHairColor,		//4		发色		ColorType		O
	ALGOBodyGesture,		//5		姿态		PostureType		O
	ALGOPersonStatus,		//6		状态		Status	PersonStatusType		O
	ALGOFaceStyle,			//7		脸型		ALGOBodyFaceStyle	FaceStyleType		O
	ALGOFacialFeature,		//8		脸部特征	FacialFeature	FacialFeatureType		O
	ALGOPhysicalFeature,	//9		体貌特征	PhysicalFeature	PhysicalFeatureType		O
	ALGOBodyFeature,		//10	体表特征	BodyFeature	BodyFeatureType		O
	ALGOHabitualAction,		//11	习惯动作	HabitualMovement	HabitualActionType		O
	ALGOBehavior,			//12	行为		Behavior	BehaviorType		O
	ALGOBehaviorDescription,//13	行为描述	BehaviorDescription	string	256	O
	ALGOAppendant,			//14	附属物		Appendant	AppendageType	128	O
	ALGOAppendantDesp,		//15	附属物描述	AppendantDescription	string	256	O
	ALGOBodyUmbrellaColor,	//16	伞颜色		ColorType		O
	ALGOBodyRespiratorColor,//17	口罩颜色	ColorType		O
	ALGOBodyCapStyle,		//18	帽子款式	HatStyleType		O
	ALGOBodyCapColor,		//19	帽子颜色	ColorType		O
	ALGOBodyGlassStyle,		//20	眼镜款式	GlassesStyleType		O
	ALGOBodyGlassColor,		//21	眼镜颜色	ColorType		O
	ALGOBodyScarfColor,		//22	围巾颜色	ColorType		O
	ALGOBodyBagStyle,		//23	包款式		BagStyleType		O
	ALGOBodyBagColor,		//24	包颜色		ColorType		O
	ALGOBodyCoatStyle,		//25	上衣款式	CoatStyleType		O
	ALGOBodyCoatLength,		//26	上衣长度	CoatLength	CoatLengthType		O
	ALGOBodyCoatColor,		//27	上衣颜色	ColorType		O
	ALGOBodyTrousersStyle,	//28	裤子款式	PantsStyleType		O
	ALGOBodyTrousersColor,	//29	裤子颜色	ColorType		O
	ALGOBodyTrousersLen,	//30	裤子长度	TrousersLen	PantsLengthType		O
	ALGOBodyShoesStyle,		//31	鞋子款式	ShoesStyleType		O
	ALGOBodyShoesColor		//32	鞋子颜色	ColorType		O
};

//人脸属性
enum ALGOFaceProperty
{
	ALGOFacePropertyPoint = 0,			//人脸关键点 左右眼  鼻子 左右嘴角
	ALGOFacePropertySkinColor,			//1		肤色	SkinColor	SkinColorType
	ALGOFacePropertyHairStyle,			//2		发型	HairStyle	HairStyleType
	ALGOFacePropertyHairColor,			//3		发色	HairColor	ColorType
	ALGOFacePropertyFaceStyle,			//4		脸型	FaceStyle	FaceStyleType
	ALGOFacePropertyFacialFeature,		//5		脸部特征	FacialFeature	FacialFeatureType
	ALGOFacePropertyPhysicalFeature,	//6		体貌特征	PhysicalFeature	PhysicalFeatureType
	ALGOFacePropertyRespiratorColor,	//7		口罩颜色	RespiratorColor	ColorType
	ALGOFacePropertyCapStyle,			//8		帽子款式	CapStyle	HatStyleType
	ALGOFacePropertyCapColor,			//9		帽子颜色	CapColor	ColorType
	ALGOFacePropertyGlassStyle,			//10	眼镜款式	GlassStyle	GlassesStyleType
	ALGOFacePropertyGlassColor			//11	眼镜颜色	GlassColor	ColorType

};

//机动车属性
enum ALGOMotorVehicleProperty
{
	ALGOMotorVehiclePropertyClass,	//车辆类型	VehicleClass	VehicleClassType		O
	ALGOMotorVehiclePropertyBrand,	//车辆品牌	VehicleBrand	VehicleBrandType		O	被标注车辆的品牌
	ALGOMotorVehiclePropertyModel,	//车辆型号	VehicleModel	VehicleModelType		O
	ALGOMotorVehiclePropertyStyles, //车辆年款	VehicleStyles	string	16	O
	ALGOMotorVehiclePropertyLength,	//车辆长度	VehicleLength	VehicleLengthType		int 5位整数，单位为毫米（mm） 
	ALGOMotorVehiclePropertyWidth,	//车辆宽度	VehicleWidth	VehicleWidthType		int 4位整数，单位为毫米（mm）
	ALGOMotorVehiclePropertyHeight,	//车辆高度	VehicleHeight	VehicleHeightType		int 4位整数，单位为毫米（mm）
	ALGOMotorVehiclePropertyColor,	//车身颜色	VehicleColor	ColorType		R
	ALGOMotorVehiclePropertyDepth	//颜色深浅	VehicleColorDepth	VehicleColorDepthType	 O
};

//GA / T16.4机动车车辆类型代码
const string ALGOVehicleClassTypeB10("B10"); //重型半挂车
const string ALGOVehicleClassTypeB11("B11"); // 重型普通半挂车
const string ALGOVehicleClassTypeB12("B12"); // 重型厢式半挂车
const string ALGOVehicleClassTypeB13("B13"); // 重型罐式半挂车
const string ALGOVehicleClassTypeB14("B14"); // 重型平板半挂车
const string ALGOVehicleClassTypeB15("B15"); // 重型集装箱半挂车
const string ALGOVehicleClassTypeB16("B16"); // 重型自卸半挂车
const string ALGOVehicleClassTypeB17("B17"); // 重型特殊结构半挂车
const string ALGOVehicleClassTypeB18("B18"); // 重型仓栅式半挂车
const string ALGOVehicleClassTypeB19("B19"); // 重型旅居半挂车
const string ALGOVehicleClassTypeB1A("B1A"); // 重型专项作业半挂车
const string ALGOVehicleClassTypeB1B("B1B"); // 重型低平板半挂车
const string ALGOVehicleClassTypeB20("B20"); // 中型半挂车
const string ALGOVehicleClassTypeB21("B21"); // 中型普通半挂车
const string ALGOVehicleClassTypeB22("B22"); // 中型厢式半挂车
const string ALGOVehicleClassTypeB23("B23"); // 中型罐式半挂车
const string ALGOVehicleClassTypeB24("B24"); // 中型平板半挂车
const string ALGOVehicleClassTypeB25("B25"); // 中型集装箱半挂车
const string ALGOVehicleClassTypeB26("B26"); // 中型自卸半挂车
const string ALGOVehicleClassTypeB27("B27"); // 中型特殊结构半挂车
const string ALGOVehicleClassTypeB28("B28"); // 中型仓栅式半挂车
const string ALGOVehicleClassTypeB29("B29"); // 中型旅居半挂车
const string ALGOVehicleClassTypeB2A("B2A"); // 中型专项作业半挂车
const string ALGOVehicleClassTypeB2B("B2B"); // 中型低平板半挂车
const string ALGOVehicleClassTypeB30("B30"); // 轻型半挂车
const string ALGOVehicleClassTypeB31("B31"); // 轻型普通半挂车
const string ALGOVehicleClassTypeB32("B32"); // 轻型厢式半挂车
const string ALGOVehicleClassTypeB33("B33"); // 轻型罐式半挂车
const string ALGOVehicleClassTypeB34("B34"); // 轻型平板半挂车
const string ALGOVehicleClassTypeB35("B35"); // 轻型自卸半挂车
const string ALGOVehicleClassTypeB36("B36"); // 轻型仓栅式半挂车
const string ALGOVehicleClassTypeB37("B37"); // 轻型旅居半挂车
const string ALGOVehicleClassTypeB38("B38"); // 轻型专项作业半挂车
const string ALGOVehicleClassTypeB39("B39"); // 轻型低平板半挂车
const string ALGOVehicleClassTypeD11("D11"); // 无轨电车
const string ALGOVehicleClassTypeD12("B12"); // 有轨电车
const string ALGOVehicleClassTypeG10("G10"); // 重型全挂车
const string ALGOVehicleClassTypeG11("G11"); // 重型普通全挂车
const string ALGOVehicleClassTypeG12("G12"); // 重型厢式全挂车
const string ALGOVehicleClassTypeG13("G13"); // 重型罐式全挂车
const string ALGOVehicleClassTypeG14("G14"); // 重型平板全挂车
const string ALGOVehicleClassTypeG15("G15"); // 重型集装箱全挂车
const string ALGOVehicleClassTypeG16("G16"); // 重型自卸全挂车
const string ALGOVehicleClassTypeG17("G17"); // 重型仓栅式全挂车
const string ALGOVehicleClassTypeG18("G18"); // 重型旅居全挂车
const string ALGOVehicleClassTypeG19("G19"); // 重型专项作业全挂车
const string ALGOVehicleClassTypeG20("G20"); // 中型全挂车
const string ALGOVehicleClassTypeG21("G21"); // 中型普通全挂车
const string ALGOVehicleClassTypeG22("G22"); // 中型厢式全挂车
const string ALGOVehicleClassTypeG23("G23"); // 中型罐式全挂车
const string ALGOVehicleClassTypeG24("G24"); // 中型平板全挂车
const string ALGOVehicleClassTypeG25("G25"); // 中型集装箱全挂车
const string ALGOVehicleClassTypeG26("G26"); // 中型自卸全挂车
const string ALGOVehicleClassTypeG27("G27"); // 中型仓栅式全挂车
const string ALGOVehicleClassTypeG28("G28"); // 中型旅居全挂车
const string ALGOVehicleClassTypeG29("G29"); // 中型专项作业全挂车
const string ALGOVehicleClassTypeG30("G30"); // 轻型全挂车
const string ALGOVehicleClassTypeG31("G31"); // 轻型普通全挂车
const string ALGOVehicleClassTypeG32("G32"); // 轻型厢式全挂车
const string ALGOVehicleClassTypeG33("G33"); // 轻型罐式全挂车
const string ALGOVehicleClassTypeG34("G34"); // 轻型平板全挂车
const string ALGOVehicleClassTypeG35("G35"); // 轻型自卸全挂车
const string ALGOVehicleClassTypeG36("G36"); // 轻型仓栅式全挂车
const string ALGOVehicleClassTypeG37("G37"); // 轻型旅居全挂车
const string ALGOVehicleClassTypeG38("G38"); // 轻型专项作业全挂车
const string ALGOVehicleClassTypeH10("H10"); // 重型货车
const string ALGOVehicleClassTypeH11("H11"); // 重型普通货车
const string ALGOVehicleClassTypeH12("H12"); // 重型厢式货车
const string ALGOVehicleClassTypeH13("H13"); // 重型封闭货车
const string ALGOVehicleClassTypeH14("H14"); // 重型罐式货车
const string ALGOVehicleClassTypeH15("H15"); // 重型平板货车
const string ALGOVehicleClassTypeH16("H16"); // 重型集装厢车
const string ALGOVehicleClassTypeH17("H17"); // 重型自卸货车
const string ALGOVehicleClassTypeH18("H18"); // 重型特殊结构货车
const string ALGOVehicleClassTypeH19("H19"); // 重型仓栅式货车
const string ALGOVehicleClassTypeH20("H20"); // 中型货车
const string ALGOVehicleClassTypeH21("H21"); // 中型普通货车
const string ALGOVehicleClassTypeH22("H22"); // 中型厢式货车
const string ALGOVehicleClassTypeH23("H23"); // 中型封闭货车
const string ALGOVehicleClassTypeH24("H24"); // 中型罐式货车
const string ALGOVehicleClassTypeH25("H25"); // 中型平板货车
const string ALGOVehicleClassTypeH26("H26"); //中型集装厢车
const string ALGOVehicleClassTypeH27("H27"); // 中型自卸货车
const string ALGOVehicleClassTypeH28("H28"); // 中型特殊结构货车
const string ALGOVehicleClassTypeH29("H29"); // 中型仓栅式货车
const string ALGOVehicleClassTypeH30("H30"); // 轻型货车
const string ALGOVehicleClassTypeH31("H31"); // 轻型普通货车
const string ALGOVehicleClassTypeH32("H32"); // 轻型厢式货车
const string ALGOVehicleClassTypeH33("H33"); // 轻型封闭货车
const string ALGOVehicleClassTypeH34("H34"); // 轻型罐式货车
const string ALGOVehicleClassTypeH35("H35"); // 轻型平板货车
const string ALGOVehicleClassTypeH37("H37"); // 轻型自卸货车
const string ALGOVehicleClassTypeH38("H38"); // 轻型特殊结构货车
const string ALGOVehicleClassTypeH39("H39"); // 轻型仓栅式货车
const string ALGOVehicleClassTypeH40("H40"); // 微型货车
const string ALGOVehicleClassTypeH41("H41"); // 微型普通货车
const string ALGOVehicleClassTypeH42("H42"); // 微型厢式货车
const string ALGOVehicleClassTypeH43("H43"); // 微型封闭货车
const string ALGOVehicleClassTypeH44("H44"); // 微型罐式货车
const string ALGOVehicleClassTypeH45("H45"); // 微型自卸货车
const string ALGOVehicleClassTypeH46("H46"); // 微型特殊结构货车
const string ALGOVehicleClassTypeH47("H47"); // 微型仓栅式货车
const string ALGOVehicleClassTypeH50("H50"); // 低速货车
const string ALGOVehicleClassTypeH51("H51"); // 普通低速货车
const string ALGOVehicleClassTypeH52("H52"); // 厢式低速货车
const string ALGOVehicleClassTypeH53("H53"); // 罐式低速货车
const string ALGOVehicleClassTypeH54("H54"); // 自卸低速货车
const string ALGOVehicleClassTypeH55("H55"); // 仓栅式低速货车
const string ALGOVehicleClassTypeJ11("J11"); // 轮式装载机械
const string ALGOVehicleClassTypeJ12("J12"); // 轮式挖掘机械
const string ALGOVehicleClassTypeJ13("J13"); // 轮式平地机械
const string ALGOVehicleClassTypeK10("K10"); // 大型客车
const string ALGOVehicleClassTypeK11("K11"); // 大型普通客车
const string ALGOVehicleClassTypeK12("K12"); // 大型双层客车
const string ALGOVehicleClassTypeK13("K13"); // 大型卧铺客车
const string ALGOVehicleClassTypeK14("K14"); // 大型铰接客车
const string ALGOVehicleClassTypeK15("K15"); // 大型越野客车
const string ALGOVehicleClassTypeK16("K16"); // 大型轿车
const string ALGOVehicleClassTypeK17("K17"); // 大型专用客车
const string ALGOVehicleClassTypeK20("K20"); // 中型客车
const string ALGOVehicleClassTypeK21("K21"); // 中型普通客车
const string ALGOVehicleClassTypeK22("K22"); // 中型双层客车
const string ALGOVehicleClassTypeK23("K23"); // 中型卧铺客车
const string ALGOVehicleClassTypeK24("K24"); // 中型铰接客车
const string ALGOVehicleClassTypeK25("K25"); // 中型越野客车
const string ALGOVehicleClassTypeK27("K27"); // 中型专用客车
const string ALGOVehicleClassTypeK30("K30"); // 小型客车
const string ALGOVehicleClassTypeK31("K31"); // 小型普通客车
const string ALGOVehicleClassTypeK32("K32"); // 小型越野客车
const string ALGOVehicleClassTypeK33("K33"); // 小型轿车
const string ALGOVehicleClassTypeK34("K34"); // 小型专用客车
const string ALGOVehicleClassTypeK40("K40"); // 微型客车
const string ALGOVehicleClassTypeK41("K41"); // 微型普通客车
const string ALGOVehicleClassTypeK42("K42"); // 微型越野客车
const string ALGOVehicleClassTypeK43("K43"); // 微型轿车
const string ALGOVehicleClassTypeM10("M10"); // 三轮摩托车
const string ALGOVehicleClassTypeM11("M11"); // 普通正三轮摩托车
const string ALGOVehicleClassTypeM12("M12"); // 轻便正三轮摩托车
const string ALGOVehicleClassTypeM13("M13"); // 正三轮载客摩托车
const string ALGOVehicleClassTypeM14("M14"); // 正三轮载货摩托车
const string ALGOVehicleClassTypeM15("M15"); // 侧三轮摩托车
const string ALGOVehicleClassTypeM20("M20"); // 二轮摩托车
const string ALGOVehicleClassTypeM21("M21"); // 普通二轮摩托车
const string ALGOVehicleClassTypeM22("M22"); // 轻便二轮摩托车
const string ALGOVehicleClassTypeN11("N11"); // 三轮汽车
const string ALGOVehicleClassTypeQ10("Q10"); // 重型牵引车
const string ALGOVehicleClassTypeQ11("Q11"); // 重型半挂牵引车
const string ALGOVehicleClassTypeQ12("Q12"); // 重型全挂牵引车
const string ALGOVehicleClassTypeQ20("Q20"); // 中型牵引车
const string ALGOVehicleClassTypeQ21("Q21"); // 中型半挂牵引车
const string ALGOVehicleClassTypeQ22("Q22"); // 中型全挂牵引车
const string ALGOVehicleClassTypeQ30("Q30"); // 轻型牵引车
const string ALGOVehicleClassTypeQ31("Q31"); // 轻型半挂牵引车
const string ALGOVehicleClassTypeQ32("Q32"); // 轻型全挂牵引车
const string ALGOVehicleClassTypeT11("T11"); // 大型轮式拖拉机
const string ALGOVehicleClassTypeT20("T20"); // 小型拖拉机
const string ALGOVehicleClassTypeT21("T21"); // 小型轮式拖拉机
const string ALGOVehicleClassTypeT22("T22"); // 手扶拖拉机
const string ALGOVehicleClassTypeT23("T23"); // 手扶变形运输机
const string ALGOVehicleClassTypeZ11("Z11"); // 大型专项作业车
const string ALGOVehicleClassTypeZ21("Z21"); // 中型专项作业车
const string ALGOVehicleClassTypeZ31("Z31"); // 小型专项作业车
const string ALGOVehicleClassTypeZ41("Z41"); // 微型专项作业车
const string ALGOVehicleClassTypeZ51("Z51"); // 重型专项作业车
const string ALGOVehicleClassTypeZ71("Z71"); // 轻型专项作业车
const string ALGOVehicleClassTypeX99("B10"); // 其他


enum ALGOVehicleBrandType
{
//0		其他
//1		大众
//2		别克
//3		宝马
//4		本田
//5		标致
//6		丰田
//7		福特
//8		日产
//9		奥迪
//10	马自达
//11	雪佛兰
//12	雪铁龙
//13	现代
//14	奇瑞
//15	起亚
//16	荣威
//17	三菱
//18	斯柯达
//19	吉利
//20	中华
//21	沃尔沃
//22	雷克萨斯
//23	菲亚特
//24	吉利帝豪
//25	东风
//26	比亚迪
//27	铃木
//28	金杯
//29	海马
//30	五菱
//31	江淮
//32	斯巴鲁
//33	英伦
//34	长城
//35	哈飞
//36	庆铃（五十铃）
//37	东南
//38	长安
//39	福田
//40	夏利
//41	奔驰
//42	一汽
//43	依维柯
//44	力帆
//45	一汽奔腾
//46	皇冠
//47	雷诺
//48	JMC
//49	MG名爵
//50	凯马
//51	众泰
//52	昌河
//53	厦门金龙
//54	上海汇众
//55	苏州金龙
//56	海格
//57	宇通
//58	中国重汽
//59	北奔重卡
//60	华菱星马汽车
//61	跃进汽车
//62	黄海汽车
//65	保时捷
//66	凯迪拉克
//67	英菲尼迪
//68	吉利全球鹰
//69	吉普
//70	路虎
//71	长丰猎豹
//73	时代汽车
//75	长安轿车
//76	陕汽重卡
//81	安凯
//82	申龙
//83	大宇
//86	中通
//87	宝骏
//88	北汽威旺
//89	广汽传祺
//90	陆风
//92	北京
//94	威麟
//95	欧宝
//96	开瑞
//97	华普
//103	讴歌
//104	启辰
//107	北汽制造
//108	纳智捷
//109	野马
//110	中兴
//112	克莱斯勒
//113	广汽吉奥
//115	瑞麟
//117	捷豹
//119	唐骏欧铃
//121	福迪
//122	莲花
//124	双环
//128	永源
//136	江南
//144	道奇
//155	大运汽车
//167	北方客车
//176	九龙
//191	宾利
//201	舒驰客车
//230	红旗
};

enum ALGOVehicleColorDepthType
{
	ALGOVehicleColorDepthType0,	//0	深
	ALGOVehicleColorDepthType1	//1	浅
};

//非机动车属性
enum ALGONoMotorVehicleProperty
{
	//车辆品牌	VehicleBrand	string	32	O	被标注车辆的品牌
	//车辆款型	VehicleType		string	64	O	被标注车辆的款式型号描述
	//车辆长度	VehicleLength	VehicleLengthType		O
	//车辆宽度	VehicleWidth	VehicleWidthType		O
	//车辆高度	VehicleHeight	VehicleHeightType		O
	//车身颜色	VehicleColor	ColorType		R	见GA / T 543.5中DE00308
};

//物品属性
enum ALGOTingProperty
{
	//物品名称	Name	string	256	R / O	被标注物品名称
	//物品形状	Shape	string	64	R / O	被标注物品形状描述
	//物品颜色	Color	ColorType		R
	//物品大小	Size	string	64	O	被标注物品大小描述
	//物品材质	Material	string	256	O	被标注物品材质描述
};

//场景属性
enum ALGOSceneProperty
{

};

//物体边界
struct ALGOObjBounding
{
	int x = 0;
	int y = 0;
	int width = 0;
	int height = 0;
};

//物体属性值的数据类型
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

//识别到的物体属性
struct ALGOObjProperty
{
	std::string propertyName;
	std::string propertyValue;
	ALGOObjPropertyValueType propertyValueType = ALGOObjPropertyTypeString;
};

//识别到的物体
struct ALGOObjectParam
{
	int objectId = 0;		//算法内部生成的对象ID,可用于判断是否为同一对象
	int objType = 0;			//物体类型
	std::string objLabel;	//标注物体的名称
	float confidence = 0;	//相识度0到1之间的浮点数，越靠近1越类似
	int roiId = 0;			//当前目标所在ROI区域ID
	ALGOObjBounding boundingBox; //物体在原图中的边界
	std::list<ALGOObjProperty> propertyList; //识别到的物体属性
};

//分析图片输入
struct ALGOImageInfo
{
	string imageId;					//输入图片的ID,由算法调用者生成，全局唯一
	ALGOImageFormat imageFormate = ALGOImageFormatJpeg; //图片格式，参见VCM_IMAGE_FORMAT_E
	int imageWidth = 0;				//输入图片的宽
	int imageHeight = 0;			//输入图片的高
	ALGOBufferType imageBufferType = ALGOBufferCPU; //存放图片的内存地址类型
	char* imageBuffer = nullptr;	//由算法调用者分配,算法调用者在algorithmVAFinished毁掉中释放
	int imageBufferLen = 0;
	char* extend = nullptr;			//扩展字段由算法调用者分配和释放
	int extendBufferLen = 0; 
};

struct ALGOVAResult
{	
	ErrAlgorithm code = ErrALGOSuccess;		//图片分析结果
	ALGOImageInfo imageInfo;				//调用AlgorithmVAInterface::analyzeImage时传入的待分析图片信息
	map<ALGOObjType, list<ALGOObjectParam>> objParams;	//分析出的图片中的物体信息,按照物体类型分组
	//list<ALGOObjectParam> objParams;		//分析出的图片中的物体信息,按照物体类型分组
	int statisticsNum = 0;					//人员、车辆等统计算法得到的统计数量。是对一系列图片统计后的数值。对objParams有做去重
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

//接口参数类型
enum ALGOInterfaceParamValueType
{
	ALGOInterfaceParamType_JPEG_STRING,
	ALGOInterfaceParamType_OPENCV_MAT,
};

// 获取接口的能力
struct ALGOAbility
{
	ALGOInterfaceParamValueType dataType;
};

enum IRMatchType
{
	IRMatchTypeLong, //长特征暴力比对,遍历目标特征，逐个比对
	IRMatchTypeShort, //短特征比对,比对聚类结果后的特征
};

struct IRFeatureInfo
{
	string imageId;	//特征对应的图片ID，全局唯一

	float featureBuf[FEATURE_MAX_SIZE] = {0}; //特征值
	int featureLen = 0; //分配的特征的内存长度
		
	float featureIndex[FEATURE_MAX_SIZE] = {0}; //特征索引首地址
	int featureIndexLen = 0; //特征索引首地址

	int quality; // 提取特征对应的质量分数，范围0~100	
};

struct IRCompareResult
{
	string srcImageId;	//特征对应的图片ID，全局唯一
	string dstImageId;	//特征对应的图片ID，全局唯一
	float similarity;	//相似度值（0-1的浮点型）
};

struct PluginParam
{
	string pluginPath;	//插件所在目录绝对路径。目录结构由算法自己决定，建议pluginPath/conf 存放插件配置文件, pluginPath/model 存放模型。
	shared_ptr<AlgoLoggerInterface> logger;
};

//Video Analysis算法分析结果回调
class AlgorithmVAListener
{
public:
	AlgorithmVAListener() {};

	virtual ~AlgorithmVAListener() {};

	//子类覆盖该函数获取分析结果
	//vaResult 算法分析的结果
	virtual void algorithmVAFinished(const std::list <ALGOVAResult>& vaResult) = 0;
};

//Video Analysis算法
class  AlgorithmVAInterface
{
public:
	AlgorithmVAInterface(){};

	virtual ~AlgorithmVAInterface() {};

	virtual ALGOAbility getAbility() = 0;

	//注册异步分析的回调监听类
	void registerAVResListener(shared_ptr<AlgorithmVAListener> vaListener) { m_vaResListener = vaListener; }

	//异步分析函数
	virtual ErrAlgorithm analyzeImageASync(const std::list<ALGOImageInfo>& imageList) = 0;

	//同步分析函数
	virtual ErrAlgorithm analyzeImageSync(const std::list<ALGOImageInfo>& imageList, std::list <ALGOVAResult>& vaResult) = 0;

	
	
protected:
	//Video Analysis分析结束后的回调监听
	weak_ptr<AlgorithmVAListener> m_vaResListener;
};

//Image Retrieval算法
class  AlgorithmIRInterface
{
public:
	AlgorithmIRInterface() {};

	virtual ~AlgorithmIRInterface() {};
	virtual ALGOAbility getAbility() = 0;

	//特征比对 1:1
	//srcFeature	比对特征
	//dstFeature	被比对特征
	//similarity	相似度值（0-1的浮点型）
	virtual ErrAlgorithm compare(const string& srcFeature, const string& dstFeature, float& similarity) = 0;

	//特征比对 N:M, N和M都可以是1
	//matchType 特征匹配类型，长特征匹配或者短特征匹配
	//srcFeature 预比对的特征列表，可以是一个特征或者多个特征
	//dstFeature 底库特征列表，M个特征
	//threshold 阈值，结果中只输出分数大于等于该值的结果(0-1浮点型)
	//limit 特征列表中每个特征比对后返回的最大结果数据量，若结果比limit小，按实际数量即可
	//result 相识度最高的几个特征值
	virtual ErrAlgorithm compare(const IRMatchType& matchType, const std::list<IRFeatureInfo>& srcFeature, const std::list<IRFeatureInfo>&  dstFeature, const float& threshold, const uint32_t& limit, std::list<IRCompareResult>& result) = 0;

	//特征提取，提取image的特征值，后面可用来做特征比对
	virtual ErrAlgorithm featureExtra(const ALGOImageInfo& imageInfo, IRFeatureInfo& fetureInfo) = 0;
};


class  AlgorithmPluginInterface
{
public:
	AlgorithmPluginInterface() {};

	virtual ~AlgorithmPluginInterface() {};

	//初始化算法插件
	virtual ErrAlgorithm pluginInitialize(const PluginParam& pluginParam, int gpuId) = 0;

	//释放算法插件
	virtual ErrAlgorithm pluginRelease() = 0;

	//创建、销毁Video Analysis算法
	//gpuId < 0 :使用CPU
	//gpuId > 0 :使用指定GPU
	virtual shared_ptr<AlgorithmVAInterface> createVAAlgorithm(int gpuId) = 0;
	virtual void destoryVAAlgorithm(shared_ptr<AlgorithmVAInterface> algo) = 0;

	//创建、销毁Image Retrieval算法
	//gpuId < 0 :使用CPU
	//gpuId > 0 :使用指定GPU
	virtual shared_ptr<AlgorithmIRInterface> createIRAlgorithm(int gpuId) = 0;
	virtual void destoryIRAlgorithm(shared_ptr<AlgorithmIRInterface> algo) = 0;
	
};