#include <opencv2/opencv.hpp>

const double PI = 3.1415926535897932;
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
std::vector<cv::Vec2f> runwayLineDetector(cv::Mat, bool writeLSDImg = false, bool writeFilteredImg = false);