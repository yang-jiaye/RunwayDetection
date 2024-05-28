/**
 * @file feature_detection.h
 * @author Yang jiaye (yjy420@sjtu.edu.cn)
 * @brief feature detection modules
 * @version 0.1
 * @date 2024-05-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#ifndef FEATURE_DETECTION_H
#define FEATURE_DETECTION_H

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>

class FeatureDetector
{
public:

    /**
     * @brief HoughCoord: hough coordinate, consisted of rho and theta
     * @param rho   distance from the origin to the line, range of rho is [-max_distance, max_distance], 
     *              sign depends on sign of intersect of line and x-axis.
     * @param theta angle between the x-axis and the normal to the line, range of theta is [-PI/2, PI/2],
     *              sign depends on the direction of the normal to the line.
     * @remark line equation: 
     *              rho = x * cos(theta) + y * sin(theta)
     */
    typedef cv::Vec2d HoughCoord;

    /**
     * @brief   Line: two end points of a line in cartesian coordinate, in the form of (x1, y1, x2, y2)
     *          LineVec: a vector of lines
     *          LineVecGrid: a grid of lines, each element is a vector of lines
     */
    typedef cv::Vec4f Line;
    typedef std::vector<Line> LineVec;
    typedef std::vector<std::vector<LineVec>> LineVecGrid;

    static constexpr double PI = 3.1415926535897932;

    /**
     * @brief detect 2 edge lines
     * 
     * @param image    image containing runway
     * @param if_save  whether to save detection result
     * @return vector of detected edge lines
     */
    static std::vector<HoughCoord> detectRunwayLine(cv::Mat image, bool if_save = false);

    /**
     * @brief detect threshold bars
     * 
     * @param image     image containing runway
     * @param lines     runway edge lines
     * @param if_save   whether to save detction result
     * @param if_debug  whether to show debug infomation
     * @return vector of dectected threshold bars
     */
    static std::vector<cv::RotatedRect> detectThreshold(cv::Mat image, std::vector<HoughCoord> lines, bool if_save = false, bool if_debug = false);

    /**
     * @brief detect bottom and upper lines
     * 
     * @param bars  detected threshold bars
     * @return vector of bottom and upper lines of threshold
     */
    static std::vector<HoughCoord> detectBottomUpperLines(const std::vector<cv::RotatedRect>& bars);

    /**
     * @brief detect slope line
     * 
     * @param LRLines left and right edge lines
     * @param BULines bottom and upper lines of threshold
     * 
     * @return HoughCoord of slope line
     */
    static HoughCoord detectSlopeLine(std::vector<HoughCoord>BULines, std::vector<HoughCoord> LRLines);

    /**
     * @brief convert line with 2 cartesian end points to hough coord
     * 
     * @param line cartesian coordinate of end points in the form of (x1, y1, x2, y2)
     * @return HoughCoord of the line
     */
    static HoughCoord computeHoughLine(Line line);

    /**
     * @brief draw line in hough coord in image and save .png
     * 
     * @param image     image to draw
     * @param line      hough coord of the line
     * @param color     color of the line
     * @param thickness thickness of the line
     */
    static void drawHoughLine(cv::Mat& image, HoughCoord line, cv::Scalar color = cv::Scalar(0, 0, 255), int thickness = 2);

private:

    static double computeLength(Line line);

    static cv::Vec2d computeIntersection(HoughCoord line1, HoughCoord line2);

    static std::vector<HoughCoord> filterLines(cv::Mat image, std::vector<Line> lines, bool if_save);

    static std::vector<cv::RotatedRect> filterThresholdBars(std::vector<cv::RotatedRect> bars, std::vector<HoughCoord> lines, bool if_save = false, cv::Mat runway_line_detect = cv::Mat(), bool if_debug = false);

    static HoughCoord fitLineFromLines(LineVec lineVec);

    static HoughCoord fitLineFromPoints(const std::vector<cv::Vec2d>& points, std::vector<double> weights = std::vector<double>(0));

};


#endif // FEATURE_DEECTION_H