#pragma once
#include<string>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<openvino/openvino.hpp>
#include <fstream>
#include <vector>
#include <random>
#include <chrono>
struct Config {
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	std::string onnx_path;
};

struct pparam_struct {
	float ratio ;
	float dw;
	float dh;
	float height;
	float width;
};

struct Object {
	cv::Rect_<float> rect;
	int              label = 0;
	float            prob = 0.0;
	cv::Mat          boxMask;
};


class YOLOV8 {
public:
	YOLOV8(Config config);
	~YOLOV8();
	void detect(cv::Mat& frame);
	std::vector<Object> get_object();

private:
	const std::vector<std::string> coconame = { "danjuan","huai_danjuan" };
	pparam_struct pparam;
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	std::vector<Object> objs;
	std::string onnx_path;
	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initialmodel();
	void preprocess_img(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float* detections,float* masks, ov::Shape& output_shape, ov::Shape& output_shape_mask);
};