# include"yolo_openvino.h"

int main(int argc, char* argv[]) {
    ov::Core core;
    std::shared_ptr<ov::Model> model = core.read_model("D:/yjs_code/yolo_openvino/yolov8_openvino/model/best_openvino_model/best.xml");
    std::vector<std::string> availableDevices = core.get_available_devices();
    std::string all_devices;
    for (auto&& device : availableDevices) {
        all_devices += device;
        all_devices += ((device == availableDevices[availableDevices.size() - 1]) ? "" : ",");
        std::cout << device;
    }
    ov::CompiledModel compileModel = core.compile_model(model, "MULTI",
        ov::device::priorities(all_devices));

    try {
        std::vector<Object> masks;
        const std::string input_model_path{ "D:/yjs_code/yolo_openvino/yolov8_openvino/model/best_openvino_model/best.xml"};
        const std::string input_folder_path{ "D:/sam2_c++/openvino_yolo/Project2/img/" };
        std::vector<cv::String> image_paths;
        cv::glob(input_folder_path + "*.jpg", image_paths, false); // 假设文件夹中只有.jpg格式的图像文件

        Config config = { 0.2,0.4,0.4,640,640, input_model_path };
        
 
        
        YOLOV8 yolomodel(config);
        int i = 0;

        for (auto image_path = image_paths.begin();image_path != image_paths.end();++image_path) {

            cv::Mat img = cv::imread(*image_path);
            yolomodel.detect(img);
            masks = yolomodel.get_object();
            std::cout << "ok";
            for (int i = 0; i < masks.size(); ++i) {
                if (masks[i].boxMask.empty())continue;
                cv::Mat canvas = cv::Mat::zeros(img.size(), CV_8UC3);
                cv::Mat roi = canvas(masks[i].rect);

                cv::Mat coloredMask;
          
                cv::cvtColor(masks[i].boxMask, coloredMask, cv::COLOR_GRAY2BGR);
                std::cout << *image_path << std::endl;
                coloredMask.convertTo(coloredMask, CV_8U, 255.0); 
                coloredMask.copyTo(roi, masks[i].boxMask);  

             
                cv::addWeighted(canvas, 0.3, img, 1.0, 0, img);

                // 绘制检测框和轮廓
                cv::rectangle(img, masks[i].rect, cv::Scalar(0, 255, 0), 2);

                // 在ROI内部绘制轮1廓
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(masks[i].boxMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
                for (auto& contour : contours) {
                    for (auto& point : contour) {
                        point.x += masks[i].rect.x;  // 坐标偏移
                        point.y += masks[i].rect.y;
                    }
                }
                cv::drawContours(img, contours, -1, cv::Scalar(0, 0, 255), 2);
            }
            std::string path_write= "D:/sam2_c++/openvino_yolo/Project2/res/" + std::to_string(i) + ".jpg";
            i++;
            cv::imwrite(path_write,img);
            //cv::namedWindow("1", cv::WINDOW_NORMAL);  // 允许调整窗口大小
            //cv::resizeWindow("1", 800, 600);          // 设置窗口大小为 800×600
            //cv::imshow("1", img);
            //cv::waitKey();
        }
    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}


//#include <iostream>
//#include <string>
//#include <vector>
//
//#include <openvino/openvino.hpp> //openvino header file
//#include <opencv2/opencv.hpp>    //opencv header file
//
//std::vector<cv::Scalar> colors = { cv::Scalar(0, 0, 255) , cv::Scalar(0, 255, 0) , cv::Scalar(255, 0, 0) ,
//                                   cv::Scalar(255, 100, 50) , cv::Scalar(50, 100, 255) , cv::Scalar(255, 50, 100) };
//const std::vector<std::string> class_names = {
//    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
//    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
//    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
//    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
//    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
//    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
//    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
//    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
//    "hair drier", "toothbrush" };
//
//using namespace cv;
//using namespace dnn;
//
//// Keep the ratio before resize
//Mat letterbox(const cv::Mat& source)
//{
//    int col = source.cols;
//    int row = source.rows;
//    int _max = MAX(col, row);
//    Mat result = Mat::zeros(_max, _max, CV_8UC3);
//    source.copyTo(result(Rect(0, 0, col, row)));
//    return result;
//}
//
//int main(int argc, char* argv[])
//{
//    // -------- Step 1. Initialize OpenVINO Runtime Core --------
//    ov::Core core;
//
//    // -------- Step 2. Compile the Model --------
//    auto compiled_model = core.compile_model("C:/Users/Admin/Documents/WeChat Files/wxid_sw6pddplsk6z22/FileStorage/File/2025-04/best_openvino_model/best_openvino_model/best.xml", "CPU");
//
//    // -------- Step 3. Create an Inference Request --------
//    ov::InferRequest infer_request = compiled_model.create_infer_request();
//
//    // -------- Step 4.Read a picture file and do the preprocess --------
//    Mat img = cv::imread("C:/Users/Admin/Documents/WeChat Files/wxid_sw6pddplsk6z22/FileStorage/File/2025-04/best_openvino_model/best_openvino_model/1.jpg");
//    // Preprocess the image
//    Mat letterbox_img = letterbox(img);
//    float scale = letterbox_img.size[0] / 640.0;
//    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);
//
//    // -------- Step 5. Feed the blob into the input node of the Model -------
//    // Get input port for model with one input
//    auto input_port = compiled_model.input();
//    // Create tensor from external memory
//    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
//    // Set input tensor for model with one input
//    infer_request.set_input_tensor(input_tensor);
//
//    // -------- Step 6. Start inference --------
//    infer_request.infer();
//
//    // -------- Step 7. Get the inference result --------
//    auto output = infer_request.get_output_tensor(0);
//    auto output_shape = output.get_shape();
//    std::cout << "The shape of output tensor:" << output_shape << std::endl;
//    int rows = output_shape[2];        //8400
//    int dimensions = output_shape[1];  //84: box[cx, cy, w, h]+80 classes scores
//
//    // -------- Step 8. Postprocess the result --------
//    float* data = output.data<float>();
//    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
//    transpose(output_buffer, output_buffer); //[8400,84]
//    float score_threshold = 0.25;
//    float nms_threshold = 0.5;
//    std::vector<int> class_ids;
//    std::vector<float> class_scores;
//    std::vector<Rect> boxes;
//
//    // Figure out the bbox, class_id and class_score
//    for (int i = 0; i < output_buffer.rows; i++) {
//        Mat classes_scores = output_buffer.row(i).colRange(4, 9);
//        Point class_id;
//        double maxClassScore;
//        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);
//
//        if (maxClassScore > score_threshold) {
//            class_scores.push_back(maxClassScore);
//            class_ids.push_back(class_id.x);
//            float cx = output_buffer.at<float>(i, 0);
//            float cy = output_buffer.at<float>(i, 1);
//            float w = output_buffer.at<float>(i, 2);
//            float h = output_buffer.at<float>(i, 3);
//
//            int left = int((cx - 0.5 * w) * scale);
//            int top = int((cy - 0.5 * h) * scale);
//            int width = int(w * scale);
//            int height = int(h * scale);
//
//            boxes.push_back(Rect(left, top, width, height));
//        }
//    }
//    //NMS
//    std::vector<int> indices;
//    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);
//
//    // -------- Visualize the detection results -----------
//    for (size_t i = 0; i < indices.size(); i++) {
//        int index = indices[i];
//        int class_id = class_ids[index];
//        rectangle(img, boxes[index], colors[class_id % 6], 2, 8);
//        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
//        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
//        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height + 5);
//        cv::rectangle(img, textBox, colors[class_id % 6], FILLED);
//        putText(img, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
//    }
//
//    namedWindow("YOLOv8 OpenVINO Inference C++ Demo", WINDOW_AUTOSIZE);
//    imshow("YOLOv8 OpenVINO Inference C++ Demo", img);
//    waitKey(0);
//    destroyAllWindows();
//    return 0;
//}