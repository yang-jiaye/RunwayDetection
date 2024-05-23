#ifndef FEATURE_DETECTION_H
#define FEATURE_DETECTION_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>

class FeatureDetector
{
public:

    typedef cv::Vec2d HoughCoord;
    typedef cv::Vec4f Line;
    typedef std::vector<Line> LineVec;
    typedef std::vector<std::vector<LineVec>> LineVecGrid;

    static constexpr double PI = 3.1415926535897932;

    // detect 2 slant lines
    static std::vector<HoughCoord> detectRunwayLine(cv::Mat, bool if_save = false);

    // detect threshold
    static std::vector<cv::RotatedRect> detectThreshold(cv::Mat image, std::vector<HoughCoord> lines, bool if_save = false, bool if_debug = false);

    // detect bottom and upper lines
    static std::vector<HoughCoord> detectBottomUpperLines(const std::vector<cv::RotatedRect>& threshs);

    // detect slope line
    static HoughCoord detectSlopeLine(std::vector<HoughCoord>BULines, std::vector<HoughCoord> LRLines);

    // convert line with 2 cartesian end points to hough coord
    static HoughCoord computeHoughLine(Line line);

    // draw line in hough coord in image
    static void drawHoughLine(cv::Mat& image, HoughCoord line, cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 2);

private:

    static double computeLength(Line line);

    static cv::Vec2d computeIntersection(HoughCoord line1, HoughCoord line2);

    static std::vector<HoughCoord> filterLines(cv::Mat image, std::vector<Line> lines, bool if_save);

    static std::vector<cv::RotatedRect> filterThresholds(std::vector<cv::RotatedRect> threshs, std::vector<HoughCoord> lines, bool if_save = false, cv::Mat runway_line_detect = cv::Mat(), bool if_debug = false);

    static HoughCoord fitLineFromLines(LineVec lineVec);

    static HoughCoord fitLineFromPoints(const std::vector<cv::Vec2d>& points, std::vector<double> weights = std::vector<double>(0));

};


#endif // FEATURE_DEECTION_H