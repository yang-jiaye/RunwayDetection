#include "lsd.h"

struct NumberIndex {
    double number;
    int index;
};

cv::Vec3f getRhoThetaLength(cv::Vec4f line)
{
    float x1 = line[0];
    float y1 = line[1];
    float x2 = line[2];
    float y2 = line[3];
    float theta = 0;
    if(x1 == x2){
        theta = PI/2;
    }
    double k = (y1 - y2) / (x1 - x2);
    theta = std::atan(k);
    float A = y2 - y1;
    float B = x2 - x1;
    float C = x2*y1 - x1*y2;
    float length = sqrt(A*A + B*B);
    float rho = abs(C) / length;
    cv::Vec3f rhoThetaLength(rho, theta, length);
    return rhoThetaLength;
}

std::vector<cv::Vec2f> runwayLineDetector(
                                            cv::Mat bgrImage, 
                                            bool writeLSDImg = false, //write LSD results
                                            bool writeFilteredImg = false //write filtered results
                                         )
{
    //=====================================================
    // 1. preprocess image

    cv::Mat grayImage;
    cv::cvtColor(bgrImage, grayImage, cv::COLOR_BGR2GRAY);

    //=====================================================
    // 2. LSD

    // create LSD object
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

    // detect line feature
    std::vector<cv::Vec4f> lines; // storage line features
    double startTime = double(cv::getTickCount());

    // use opencv lsd implementation here
    lsd->detect(grayImage, lines);

    double endTime = double(cv::getTickCount());
    double durationTime = (endTime - startTime) * 1000 / cv::getTickFrequency();
    std::cout << "==================It took " << durationTime << " ms to detect line features.==================" << std::endl<<std::endl;

    if(writeLSDImg){
        cv::Mat drawImage = grayImage;
        lsd->drawSegments(drawImage, lines);
        cv::imwrite("./originalImage.jpg", drawImage);
    }

    /*
    the most difficult part is to find specific 3 lines from 
    all the detected lines above. We filter out short line 
    below threshould, and rotation angle lower threshould first. 
    Find the 2 slope interval which has the largest 
    accumulating length. Use mean theta and d to find the 2 slant 
    runway lines. Between the 2 lines repeat the procedure above
    to find the bottom line.
    */
    
    //=====================================================
    // 3 filter out lines and group them

    // in filteredLines, each Vec4f storage coords of a line
    std::vector<cv::Vec4f> filteredLines;
    std::vector<cv::Vec4f>::iterator lineIter; // iterate throughout lines_std
    float rho, theta, length;
    double lengthThreshold = 10; //length threshold
    double angleThreshold = PI/4; //angle threshold
    for(lineIter = lines.begin(); lineIter != lines.end(); ++lineIter){
        cv::Vec3f rtl = getRhoThetaLength(*lineIter);
        theta = rtl[1];
        length = rtl[2];
        if(
            length > lengthThreshold 
            && abs(theta) > angleThreshold
        ){
            filteredLines.push_back(*lineIter);
        }
    }
    
    if(writeFilteredImg){
        cv::Mat drawImage = grayImage;
        cv::cvtColor(drawImage, drawImage, cv::COLOR_BGR2RGB);
        for (auto l:filteredLines) {
            cv::line(drawImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        }
        cv::imwrite("../filteredImage.jpg", drawImage);
    }

    //=====================================================
    std::vector<cv::Vec2f> result;

    //=====================================================
    // 7 detect bottom line
    // To do use K means to find the bottom line


    return result;
}