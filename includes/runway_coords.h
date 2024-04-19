#ifndef RUNWAY_COORDS_H
#define RUNWAY_COORDS_H

#include <opencv2/opencv.hpp>

// 4 corner points of the runway
cv::Vec3f runway_P1(0, 22.5, 0);
cv::Vec3f runway_P2(1891, 22.5, 0);
cv::Vec3f runway_P3(1891, -22.5, 0);
cv::Vec3f runway_P4(0, -22.5, 0);

// 4 corner points of threshold area
cv::Vec3f thresh_P1(69.2, 22.5, 0);
cv::Vec3f thresh_P2(99.2, 22.5, 0);
cv::Vec3f thresh_P3(99.2, -22.5, 0);
cv::Vec3f thresh_P4(69.2, -22.5, 0);

// 12 thresholds coords
// x1, y1 is the left-up corner, x2, y2 is the right-down corner
std::vector<std::vector<cv::Vec3f>> get_thresholds_coords()
{
    std::vector<std::vector<cv::Vec3f>> thresholds_coords;
    for(int i = 0; i < 6; i++)
    {
        std::vector<cv::Vec3f> threshold;
        float x1 = 99.2;
        float y1 = -21.6435 + 1.8 * (2 * i);
        float x2 = 69.2;
        float y2 = -21.6435 + 1.8 * (2 * i + 1);
        threshold.push_back(cv::Vec3f(x1, y1, 0));
        threshold.push_back(cv::Vec3f(x2, y1, 0));
        threshold.push_back(cv::Vec3f(x2, y2, 0));
        threshold.push_back(cv::Vec3f(x1, y2, 0));
        thresholds_coords.push_back(threshold);
    }

    for(int i = 0; i < 6; i++)
    {
        std::vector<cv::Vec3f> threshold;
        float x1 = 99.2;
        float y1 = 1.8509 + 1.8 * (2 * i);
        float x2 = 69.2;
        float y2 = 1.8509 + 1.8 * (2 * i + 1);
        threshold.push_back(cv::Vec3f(x1, y1, 0));
        threshold.push_back(cv::Vec3f(x2, y1, 0));
        threshold.push_back(cv::Vec3f(x2, y2, 0));
        threshold.push_back(cv::Vec3f(x1, y2, 0));
        thresholds_coords.push_back(threshold);
    }

    return thresholds_coords;
}

#endif // RUNWAY_COORDS_H