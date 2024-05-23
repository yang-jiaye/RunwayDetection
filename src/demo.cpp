#include "feature_detection.h"

typedef FeatureDetector::HoughCoord HoughCoord;
typedef FeatureDetector::Line Line;

int main()
{
    // read image
    std::string path = "../images/4.png";
    cv::Mat image = cv::imread(path);

    // detect runway lines
    std::cout<<"Detecting runway lines..."<<std::endl;
    auto LRLines = FeatureDetector::detectRunwayLine(image, false);

    for(auto line: LRLines){
        FeatureDetector::drawHoughLine(image, line, cv::Scalar(0, 255, 0), 2);
    }

    // detect threshold
    std::cout<<"Detecting runway thresholds..."<<std::endl;
    auto threshs = FeatureDetector::detectThreshold(image, LRLines, false, false);
    for(auto thresh: threshs){
        cv::Point2f vertices[4];
        thresh.points(vertices);
        for(int i = 0; i < 4; i++){
            cv::line(image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 2);
        }
    }

    // detect bottom and upper lines
    std::cout<<"Detecting bottom and upper lines..."<<std::endl;
    auto BULines = FeatureDetector::detectBottomUpperLines(threshs);
    for(auto line: BULines){
        FeatureDetector::drawHoughLine(image, line, cv::Scalar(255, 0, 0), 2);
    }

    // detect slope line
    std::cout<<"Detecting slope line..."<<std::endl;
    auto slopeLine = FeatureDetector::detectSlopeLine(BULines, LRLines);
    FeatureDetector::drawHoughLine(image, slopeLine, cv::Scalar(255, 255, 0), 2);

    // save image
    cv::imwrite("./output.png", image);

    return 0;
}