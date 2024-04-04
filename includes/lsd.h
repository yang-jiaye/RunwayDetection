#include <opencv2/opencv.hpp>
#include <algorithm>
#include <Eigen/Dense>

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
std::vector<cv::Vec2f> runwayLineDetector(cv::Mat, bool writeImage = false);
std::vector<cv::Vec2f> houghTransform(cv::Mat bgrImage, std::vector<cv::Vec4f> lines, bool writeImage);
std::vector<cv::Vec2f> houghTransformM(cv::Mat bgrImage, cv::Mat mask, bool writeImage = false);
HoughCoord cartesianToHough(cv::Vec4f line);
cv::Vec2f computeResults(LineVec lineVec);
