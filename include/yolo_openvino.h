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

struct Obsegment{
	cv::Rect box;
	int Class_id;
	float Score;
	cv::Mat Mask;
};


class YOLOV8 {
public:
	YOLOV8(Config config);
	~YOLOV8();
	void detect(cv::Mat& frame);
	std::vector<Obsegment> get_mask();

private:
	cv::Size Origin_img_size;
	int Origin_img_type;
	float confThreshold;
	float nmsThreshold;
	float scoreThreshold;
	int inpWidth;
	int inpHeight;
	float rxy;   // the width ratio of original image and resized image
	std::vector<Obsegment> Obsegments;

	std::string onnx_path;

	ov::Tensor input_tensor;
	ov::InferRequest infer_request;
	ov::CompiledModel compiled_model;
	void initialmodel();
	void preprocess_img(cv::Mat& frame);
	void postprocess_img(cv::Mat& frame, float* detections,float* masks, ov::Shape& output_shape, ov::Shape& output_shape_mask);
};