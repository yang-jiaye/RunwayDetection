#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include "lsd.h"

void processImage(cv::Mat runway_line_detect)
{
    std::vector<cv::Vec2f> LRLines = detectRunwayLine(runway_line_detect);

    for(auto line: LRLines){
        float rho = line[0];
        float theta = line[1];
        // std::cout<<rho<<" "<<theta<<std::endl;
        float x0 = rho * std::cos(theta);
        float y0 = rho * std::sin(theta);
        float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
        float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
        float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
        float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
        cv::line(runway_line_detect, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    }

    std::vector<cv::RotatedRect> threshs = detectRunwayThreshold(runway_line_detect, LRLines);
    for(auto thresh: threshs){
        cv::Point2f vertices[4];
        thresh.points(vertices);
        for(int i = 0; i < 4; i++){
            cv::line(runway_line_detect, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 2);
        }
    }

    auto BULines = detectBottomAndUpperLines(threshs);
    for(auto line: BULines){
        float rho = line[0];
        float theta = line[1];
        // std::cout<<rho<<" "<<theta<<std::endl;
        float x0 = rho * std::cos(theta);
        float y0 = rho * std::sin(theta);
        float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
        float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
        float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
        float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
        cv::line(runway_line_detect, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
    }

    auto intersects = detectSlopeLine(BULines, LRLines);
    cv::line(runway_line_detect, cv::Point(intersects[0][0], intersects[0][1]), cv::Point(intersects[1][0], intersects[1][1]), cv::Scalar(255, 255, 0), 2);
}

void playAnimation(const std::string& folderPath, int interval) {
    std::vector<std::string> imageFiles;
    cv::glob(folderPath, imageFiles);

    for (const auto& imageFile : imageFiles) {
        std::cout<<"[playAnimation]show image"<<imageFile<<std::endl;
        cv::Mat image = cv::imread(imageFile);

        processImage(image);
        
        cv::imshow("Animation", image);
        cv::waitKey(interval);
    }

    cv::destroyAllWindows();
}

int main() {
    std::string folderPath = "../data/mav0/cam0/data";
    int interval = 10;

    playAnimation(folderPath, interval);

    return 0;
}
