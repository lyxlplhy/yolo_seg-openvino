# include"yolo_openvino.h"

int main(int argc, char* argv[]) {

    try {
        std::vector<Obsegment> masks;
        const std::string input_model_path{ "D:/sam2/ultralytics-main/ultralytics-main/model/danjuan-seg/best_n_openvino_model/best_n.xml"};
        const std::string input_image_path{"D:/sam2/ultralytics-main/ultralytics-main/model/danjuan-seg/b10aa679ca73a31c1a5e2225bb3ffdd.png"};
        Config config = { 0.2,0.4,0.4,640,640, input_model_path };
        
        cv::Mat img = cv::imread(input_image_path);
        YOLOV8 yolomodel(config);
        while (1) {
            auto start = std::chrono::system_clock::now();
            yolomodel.detect(img);
            masks = yolomodel.get_mask();
            auto end = std::chrono::system_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
            std::cout << "执行总时间时间: " << duration << " 毫秒" << std::endl;
        }
        for (int i = 0;i < masks.size();++i)
        {
            cv::imshow("1", masks[i].Mask);
            cv::waitKey();
        }

    }
    catch (const std::exception& ex) {
        std::cerr << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;

}

