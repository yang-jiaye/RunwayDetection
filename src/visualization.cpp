#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include "feature_detection.h"

typedef FeatureDetector::HoughCoord HoughCoord;
typedef FeatureDetector::Line Line;

void processImage(cv::Mat image)
{
    // detect runway lines
    auto LRLines = FeatureDetector::detectRunwayLine(image, false);
    for(auto line: LRLines){
        FeatureDetector::drawHoughLine(image, line, cv::Scalar(0, 255, 0), 2);
    }

    // detect threshold
    auto threshs = FeatureDetector::detectThreshold(image, LRLines, false, false);
    for(auto thresh: threshs){
        cv::Point2f vertices[4];
        thresh.points(vertices);
        for(int i = 0; i < 4; i++){
            cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 2);
        }
    }

    // detect bottom and upper lines
    auto BULines = FeatureDetector::detectBottomUpperLines(threshs);
    for(auto line: BULines){
        FeatureDetector::drawHoughLine(image, line, cv::Scalar(255, 0, 0), 2);
    }

    // detect slope line
    auto slopeLine = FeatureDetector::detectSlopeLine(BULines, LRLines);
    FeatureDetector::drawHoughLine(image, slopeLine, cv::Scalar(255, 255, 0), 2);
}

void playAnimation(const std::string& folderPath, int interval) {
    std::vector<std::string> imageFiles;
    cv::glob(folderPath, imageFiles);

    for (const auto& imageFile : imageFiles) {
        std::cout<<"[playAnimation]:\tshow image"<<imageFile<<std::endl;
        cv::Mat image = cv::imread(imageFile);

        processImage(image);
        
        cv::imshow("Animation", image);
        cv::waitKey(interval);
    }

    cv::destroyAllWindows();
}

int main() {
    std::string folderPath = "../data/EuRoc_FAIM_20240401/mav0/cam0/data";
    int interval = 10;

    playAnimation(folderPath, interval);

    return 0;
}