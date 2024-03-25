#include "includes/lsd.h"

int main()
{
    std::string path = "../demo.jpg";
    cv::Mat img = cv::imread(path);

    std::vector<cv::Vec2f> lines = runwayLineDetector(img);

    for(auto line: lines){
        float d = line[0];
        float theta = line[1];
        std::cout<<d<<" "<<theta<<std::endl;
        float x0 = d * std::sin(theta);
        float y0 = -d * std::cos(theta);
        float x1 = static_cast<int>(x0 + 4000 * (std::cos(theta)));
        float y1 = static_cast<int>(y0 + 4000 * (std::sin(theta)));
        float x2 = static_cast<int>(x0 - 1000 * (std::cos(theta)));
        float y2 = static_cast<int>(y0 - 1000 * (std::sin(theta)));
        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 20);
    }

    cv::imwrite("../output.jpg", img);
    // cv::namedWindow("Image", cv::WINDOW_NORMAL);
    // cv::imshow("Image", img);
    // cv::resizeWindow("Image", 800, 600);
    // cv::waitKey(0);
    return 0;
}