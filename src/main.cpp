#include "lsd.h"

int main()
{
    std::string path = "../images/1711643355411280128.png";
    cv::Mat img = cv::imread(path);
    auto runway_line_detect = img.clone();

    int width = img.cols;
    int height = img.rows;
    int Bxmin = width/4;
    int Bxmax = 3*width/4;
    int Bymin = 0;
    int Bymax = height;
    cv::Rect roi(Bxmin, Bymin, Bxmax - Bxmin, Bymax - Bymin);
    cv::Mat cropped = img(roi);

    std::cout<<"Detecting runway lines..."<<std::endl;
    std::vector<cv::Vec2f> LRLines = detectRunwayLine(cropped, true);

    for(auto line: LRLines){
        float rho = line[0];
        float theta = line[1];
        std::cout<<rho<<" "<<theta<<std::endl;
        rho += Bxmin * std::cos(theta) + Bymin * std::sin(theta);

        float x0 = rho * std::cos(theta);
        float y0 = rho * std::sin(theta);
        float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
        float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
        float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
        float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
        cv::line(runway_line_detect, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
    }

    std::cout<<"Detecting runway thresholds..."<<std::endl;
    std::vector<cv::RotatedRect> threshs = detectRunwayThreshold(cropped, LRLines, true);
    for(auto thresh: threshs){
        cv::Point2f vertices[4];
        thresh.points(vertices);
        for(int i = 0; i < 4; i++){
            vertices[i].x += Bxmin;
        }
        for(int i = 0; i < 4; i++){
            cv::line(runway_line_detect, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 0, 255), 2);
        }
    }

    std::cout<<"Detecting bottom and upper lines..."<<std::endl;
    auto BULines = detectBottomAndUpperLines(threshs);
    for(auto line: BULines){
        float rho = line[0];
        float theta = line[1];
        // std::cout<<rho<<" "<<theta<<std::endl;
        rho += Bxmin * std::cos(theta) + Bymin * std::sin(theta);

        float x0 = rho * std::cos(theta);
        float y0 = rho * std::sin(theta);
        float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
        float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
        float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
        float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
        cv::line(runway_line_detect, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255, 0, 0), 2);
    }


    std::cout<<"Detecting slope line..."<<std::endl;
    auto intersects = detectSlopeLine(BULines, LRLines);
    cv::line(runway_line_detect, cv::Point(intersects[0][0] + Bxmin, intersects[0][1]), cv::Point(intersects[1][0] + Bxmin, intersects[1][1]), cv::Scalar(255, 255, 0), 2);

    cv::imwrite("./output.png", runway_line_detect);
    return 0;
}