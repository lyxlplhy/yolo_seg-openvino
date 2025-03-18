#include"yolo_openvino.h"
#include<iostream>
#include<string>
#include<time.h>

const std::vector<std::string> coconame = {"danjuan"};
cv::Mat letterbox(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}
float sigmoid_function(float a) {
    float b = 1. / (1. + exp(-a));
    return b;
}


YOLOV8::YOLOV8(Config config) {
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->scoreThreshold = config.scoreThreshold;
    this->inpWidth = config.inpWidth;
    this->inpHeight = config.inpHeight;
    this->onnx_path = config.onnx_path;
    this->initialmodel();

}
YOLOV8::~YOLOV8() {}
void YOLOV8::detect(cv::Mat& frame) {

    preprocess_img(frame);
   
    const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);
    const ov::Tensor& output_tensor_mask = infer_request.get_output_tensor(1);
    ov::Shape output_shape_dectect = output_tensor.get_shape();
    ov::Shape output_shape_mask = output_tensor_mask.get_shape();
    float* detections = output_tensor.data<float>();
    float* masks = output_tensor_mask.data<float>();

    auto start_infer = std::chrono::system_clock::now();
   
    this->postprocess_img(frame, detections, masks, output_shape_dectect, output_shape_mask);
    auto end_infer = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count() / 1000.0;
    // std::cout << "afte: " << duration << " ms" << std::endl;
    

}



void YOLOV8::initialmodel() {
    ov::Core core;
    this->compiled_model = core.compile_model(this->onnx_path, "CPU");
    this->infer_request = compiled_model.create_infer_request();
}


void YOLOV8::preprocess_img(cv::Mat& frame) {
    auto start_img = std::chrono::system_clock::now();
    this->Origin_img_size = frame.size();
    this->Origin_img_type = frame.type();
    cv::Mat letterbox_img = letterbox(frame);
    this->rxy = letterbox_img.size[0] / 640.0;
    cv::Mat blob = cv::dnn::blobFromImage(letterbox_img, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true);
    auto input_port = this->compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);
    auto end_img = std::chrono::system_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_img - start_img).count() / 1000.0;
    // std::cout << "befor: " << duration << " ms" << std::endl;

    auto start_infer = std::chrono::system_clock::now();
    infer_request.infer();//
    auto end_infer = std::chrono::system_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count() / 1000.0;
    // std::cout << "inf: " << duration1 << " ms" << std::endl;
}


void YOLOV8::postprocess_img(cv::Mat& frame, float* detections, float* mask, ov::Shape& output_shape, ov::Shape& output_shape_mask) {
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, detections);
    cv::Mat proto(32, 25600, CV_32F, mask); //[32,25600]
    transpose(output_buffer, output_buffer); //[8400,116]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector< cv::Rect> boxes;
    std::vector< cv::Mat> mask_confs;
    this->Obsegments.clear();
    for (int i = 0; i < output_buffer.rows; i++) {
        cv::Mat classes_scores = output_buffer.row(i).colRange(4, 4+coconame.size());
        cv::Point class_id;
        double maxClassScore = 0;

        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * this->rxy);
            int top = int((cy - 0.5 * h) * this->rxy);
            int width = int(w * this->rxy);
            int height = int(h * this->rxy);

            cv::Mat mask_conf = output_buffer.row(i).colRange(4 + coconame.size(), 36+ coconame.size());
            mask_confs.push_back(mask_conf);
            boxes.push_back(cv::Rect(left, top, width, height));
        }
    }
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);
    //cv::Mat rgb_mask = cv::Mat::zeros(this->Origin_img_size, this->Origin_img_type);
    //cv::Mat masked_img;
    //cv::RNG rng;
    for (auto index = indices.begin();index != indices.end();++index)
    {
        int class_id = *index;

        cv::Mat m = mask_confs[*index] * proto;
        /*for (int col = 0; col < m.cols; col++) {
            m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
        }*/
        if (m.isContinuous() && m.type() == CV_32F) {
            float* ptr = m.ptr<float>(0);  
            const int cols = m.cols;
            for (int col = 0; col < cols; ++col) {
                ptr[col] = sigmoid_function(ptr[col]);
            }
        }
        cv::Mat m1 = m.reshape(1, 160); // 1x25600 -> 160x160
        int x1 = std::max(0, boxes[*index].x);
        int y1 = std::max(0, boxes[*index].y);
        int x2 = std::max(0, boxes[*index].br().x);
        int y2 = std::max(0, boxes[*index].br().y);
        int mx1 = int(x1 / this->rxy * 0.25);
        int my1 = int(y1 / this->rxy * 0.25);
        int mx2 = int(x2 / this->rxy * 0.25);
        int my2 = int(y2 / this->rxy * 0.25);
        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;
        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

      /*  for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 255.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }*/
        if (rm.isContinuous() && rm.type() == CV_32F) {
            float* p = rm.ptr<float>(0);
            const int total = rm.rows * rm.cols;
            for (int i = 0; i < total; ++i) {
                p[i] = (p[i] > 0.5f) ? 255.0f : 0.0f;
            }
        }
       /* cv::parallel_for_(cv::Range(0, rm.rows), [&](const cv::Range& range) {
            for (int r = range.start; r < range.end; ++r) {
                float* p = rm.ptr<float>(r);
                for (int c = 0; c < rm.cols; ++c) {
                    p[c] = p[c] > 0.5f ? 255.0f : 0.0f;
                }
            }
            });*/

        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= Origin_img_size.height) {
            y2 = Origin_img_size.height - 1;
        }
        if ((x1 + det_mask.cols) >= Origin_img_size.width) {
            x2 = Origin_img_size.width - 1;
        }
        cv::Mat mask = cv::Mat::zeros(cv::Size(Origin_img_size.width, Origin_img_size.height), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));

        Obsegment Obsegment;
        Obsegment.box = boxes[*index];
        Obsegment.Class_id = *index;
        Obsegment.Score = class_scores[*index];
        Obsegment.Mask = mask;
        this->Obsegments.push_back(Obsegment);
    }
}


std::vector<Obsegment> YOLOV8:: get_mask() {
    return  this->Obsegments;
}