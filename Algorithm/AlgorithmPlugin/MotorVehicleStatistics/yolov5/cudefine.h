#pragma once

namespace Yolo
{
	static constexpr int CHECK_COUNT = 3;
	static constexpr float IGNORE_THRESH = 0.1f;
	struct YoloKernel
	{
		int width;
		int height;
		float anchors[CHECK_COUNT * 2];
	};
	static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
	static constexpr int CLASS_NUM = 80;
	static constexpr int INPUT_H = 608;
	static constexpr int INPUT_W = 608;
	static constexpr int INPUT_C = 3;

	static constexpr int LOCATIONS = 4;
	struct alignas(float) Detection {
		//center_x center_y w h
		float bbox[LOCATIONS];
		float conf;  // bbox_conf * cls_conf
		float class_id;
	};
}