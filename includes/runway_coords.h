#ifndef RUNWAY_COORDS_H
#define RUNWAY_COORDS_H

#include <opencv2/opencv.hpp>

// 4 corner points of the runway
cv::Vec3f P1(0, 22.5, 0);
cv::Vec3f P2(1891, 22.5, 0);
cv::Vec3f P3(1891, -22.5, 0);
cv::Vec3f P4(0, -22.5, 0);

// 4 corner points of threshold area
cv::Vec3f P1(69.2, 22.5, 0);
cv::Vec3f P2(99.2, 22.5, 0);
cv::Vec3f P3(99.2, -22.5, 0);
cv::Vec3f P4(69.2, -22.5, 0);

// 12 thresholds coords
// each Vec4f restore x1, y1, x2, y2 of threshold, where x1, y1 is the left-up corner, x2, y2 is the right-down corner
// as the threshold is parallel to x-axis, left-down and right-up corner coords is x1, y2, and x2, y1
std::vector<cv::Vec4f> get_thresholds_coords()
{
    std::vector<cv::Vec4f> thresholds_coords;
    for(int i = 0; i < 6; i++)
    {
        cv::Vec4f threshold;
        threshold[0] = 99.2;
        threshold[1] = -21.6435 + 1.8 * (2 * i);
        threshold[2] = 69.2;
        threshold[3] = -21.6435 + 1.8 * (2 * i + 1);
        thresholds_coords.push_back(threshold);
    }

    for(int i = 0; i < 6; i++)
    {
        cv::Vec4f threshold;
        threshold[0] = 99.2;
        threshold[1] = 1.8509 + 1.8 * (2 * i);
        threshold[2] = 69.2;
        threshold[3] = 1.8509 + 1.8 * (2 * i + 1);
        thresholds_coords.push_back(threshold);
    }
}

#endif // RUNWAY_COORDS_H