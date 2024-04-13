#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
// #include <math.h>

const double PI = 3.1415926535897932;
typedef cv::Vec3d HoughCoord;
typedef std::vector<cv::Vec3d> HoughCoordVec;
typedef std::vector<cv::Vec4f> LineVec;
typedef std::vector<std::vector<LineVec>> LineGrid;

/*
runway line detection function
    Input: 
        image in BGR color space
    Output: 
        vector 3 Eigen::Vector2d
        Each row represents a lie [rho, theta] 
        where rho is the distance from line to origin
        theta is the slope angle of line.
        First 2 lines are slant runway lines
        The third line is the bottom line
*/
// detect 2 slant lines
std::vector<cv::Vec2f> detectRunwayLine(cv::Mat, bool writeImage = false);
std::vector<cv::Vec2f> houghTransform(cv::Mat bgrImage, std::vector<cv::Vec4f> lines, bool writeImage);
std::vector<cv::Vec2f> houghTransformM(cv::Mat bgrImage, cv::Mat mask, bool writeImage = false);

// detect threshold
std::vector<cv::RotatedRect> detectRunwayThreshold(cv::Mat bgrImage, std::vector<cv::Vec2f> lines, bool writeImage = false, bool if_debug = false);
std::vector<cv::RotatedRect> filterThresholds(std::vector<cv::RotatedRect> threshs, std::vector<cv::Vec2f> lines, bool writeImage = false, cv::Mat runway_line_detect = cv::Mat(), bool if_debug = false);

//detect bottom and upper lines
std::vector<cv::Vec2f> detectBottomAndUpperLines(const std::vector<cv::RotatedRect>& threshs);

//detect slope line
std::vector<cv::Vec2f> detectSlopeLine(std::vector<cv::Vec2f>BULines, std::vector<cv::Vec2f> LRLines);