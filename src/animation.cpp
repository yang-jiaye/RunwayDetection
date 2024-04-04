#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include "lsd.h"

void playAnimation(const std::string& folderPath, int interval) {
    std::vector<std::string> imageFiles;
    cv::glob(folderPath, imageFiles);

    for (const auto& imageFile : imageFiles) {
        std::cout<<"===============show image"<<imageFile<<std::endl;
        cv::Mat image = cv::imread(imageFile);

        std::vector<cv::Vec2f> lines = runwayLineDetector(image, true);

        for(auto line: lines){
            float rho = line[0];
            float theta = line[1];
            std::cout<<rho<<" "<<theta<<std::endl;
            float x0 = rho * std::cos(theta);
            float y0 = rho * std::sin(theta);
            float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
            float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
            float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
            float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
            cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 20);
        }
        
        cv::imshow("Animation", image);
        cv::waitKey(interval);
    }

    cv::destroyAllWindows();
}

int main() {
    std::string folderPath = "../data/EuRoc_FAIM_20240401/cam0/data_rgb";
    int interval = 10;

    playAnimation(folderPath, interval);

    return 0;
}
