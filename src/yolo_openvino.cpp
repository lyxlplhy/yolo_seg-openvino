#include"yolo_openvino.h"

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
    
    //preprocess and inference
    preprocess_img(frame);

    //get tenor
    const ov::Tensor& output_tensor = infer_request.get_output_tensor(0);//box_tenor
    const ov::Tensor& output_tensor_mask = infer_request.get_output_tensor(1);//mask_tensor
    ov::Shape output_shape_dectect = output_tensor.get_shape();
    ov::Shape output_shape_mask = output_tensor_mask.get_shape();
    float* detections = output_tensor.data<float>();
    float* masks = output_tensor_mask.data<float>();

    //postprocess
    this->postprocess_img(frame, detections, masks, output_shape_dectect, output_shape_mask);
}

void YOLOV8::initialmodel() {
    ov::Core core;
    this->compiled_model = core.compile_model(this->onnx_path, "CPU");//intel cpu!!!
    this->infer_request = compiled_model.create_infer_request();
}

void YOLOV8::preprocess_img(cv::Mat& frame)
{
    const float inp_h = 640;
    const float inp_w = 640;
    float       height = frame.rows;
    float       width = frame.cols;
    float r = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);
    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(frame, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = frame.clone();
    }
    float dw = inp_w - padw;
    float dh = inp_h - padh;
    dw /= 2.0f;
    dh /= 2.0f;
    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));

    //infernence
    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, { 114, 114, 114 });
    cv::Mat blob = cv::dnn::blobFromImage(tmp, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(0,0,0), true, false, CV_32F);
    auto input_port = this->compiled_model.input();
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();

    this->pparam.ratio = 1 / r;
    this->pparam.dw = dw;
    this->pparam.dh = dh;
    this->pparam.height = height;
    this->pparam.width = width;

}



void YOLOV8::postprocess_img(cv::Mat& frame, float* detections, float* mask, ov::Shape& output_shape, ov::Shape& output_shape_mask)
{
    this->objs.clear();
    cv::Mat output_buffer(output_shape[1], output_shape[2], CV_32F, detections);
    output_buffer = output_buffer.t();
    cv::Mat protos(32, 25600, CV_32F, mask); //[32,25600]
   
    int num_classes = this->coconame.size();
    std::vector<int>      labels;
    std::vector<float>    scores;
    std::vector<cv::Rect> bboxes;
    std::vector<cv::Mat>  mask_confs;
    std::vector<int>      indices;
    auto& dw = this->pparam.dw;
    auto& dh = this->pparam.dh;
    auto& width = this->pparam.width;
    auto& height = this->pparam.height;
    auto& ratio = this->pparam.ratio;

    for (int i = 0; i < output_buffer.rows; i++) {
        auto  row_ptr = output_buffer.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  mask_confs_ptr = row_ptr + 4 + num_classes;
        auto  max_s_ptr = std::max_element(scores_ptr, scores_ptr + num_classes);
        float score = *max_s_ptr;
        if (score > 0.5f) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = std::clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = std::clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = std::clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = std::clamp((y + 0.5f * h) * ratio, 0.f, height);

            int label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            cv::Mat mask_conf = cv::Mat(1, 32, CV_32F, mask_confs_ptr);
            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
            mask_confs.push_back(mask_conf);
        }
    }
    cv::dnn::NMSBoxes(bboxes, scores, 0.5, 0.65, indices);
    cv::Mat masks;
    int cnt = 0;
    int topk = 100;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        cv::Rect tmp = bboxes[i];
        Object   obj;
        obj.label = labels[i];
        obj.rect = tmp;
        obj.prob = scores[i];
        masks.push_back(mask_confs[i]);
        this->objs.push_back(obj);
        cnt += 1;
    }
    if (!masks.empty()) {
        cv::Mat matmulRes = (masks * protos).t();
        cv::Mat maskMat = matmulRes.reshape(indices.size(), { 160, 160 });//yolo_mask shape

        std::vector<cv::Mat> maskChannels;
        cv::split(maskMat, maskChannels);
        int scale_dw = dw / 640 * 160;//640 is yolo input shape
        int scale_dh = dh / 640 * 160;

        cv::Rect roi(scale_dw, scale_dh, 160 - 2 * scale_dw, 160 - 2 * scale_dh);

        for (int i = 0; i < indices.size(); i++) {
            cv::Mat dest, mask;
            cv::exp(-maskChannels[i], dest);
            dest = 1.0 / (1.0 + dest);
            dest = dest(roi);
            cv::resize(dest, mask, cv::Size((int)width, (int)height), cv::INTER_LINEAR);
            objs[i].boxMask = mask(objs[i].rect) > 0.5f;
           /* cv::imshow("1", objs[i].boxMask);
            cv::waitKey();*/
        }
    }
}

std::vector<Object> YOLOV8::get_object()
{
    return this->objs;
}