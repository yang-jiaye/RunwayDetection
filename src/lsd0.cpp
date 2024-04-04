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
                                            bool writeLSDres = false //write LSD results
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

    if(writeLSDres){
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

    // in filteredLines, each Vec3f storage rho theta and length of a line
    std::vector<cv::Vec3f> filteredLines;
    std::vector<cv::Vec4f> filtered4FLines = lines;
    filtered4FLines.clear();
    std::vector<cv::Vec4f>::iterator lineIter; // iterate throughout lines_std
    float rho, theta, length;
    double lengthThreshold = 10; //length threshold
    double angleThreshold = PI/4; //angle threshold
    for(lineIter = lines.begin(); lineIter != lines.end(); ++lineIter){
        // filtered4FLines.push_back(*lineIter);
        cv::Vec3f rtl = getRhoThetaLength(*lineIter);
        theta = rtl[1];
        length = rtl[2];
        if(
            length > lengthThreshold 
            && abs(theta) > angleThreshold
        ){
            filteredLines.push_back(rtl);
            filtered4FLines.push_back(*lineIter);
        }
    }
    
    cv::Mat drawImage = grayImage;
    cv::cvtColor(drawImage, drawImage, cv::COLOR_BGR2RGB);
    for (auto l:filtered4FLines) {
        cv::line(drawImage, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
    }

    std::cout<<filteredLines.size()<<filtered4FLines.size()<<std::endl;
    cv::imwrite("../filteredImage.jpg", drawImage);

    //=====================================================
    // 4 accumulate total length in each bin

    int nBins = 20; // 20 bins
    double intervalBins = PI/nBins;
    std::vector<std::vector<cv::Vec3f>> bins(nBins); // nBins interval, each interval storage several Vec3f inside
    std::vector<double> binLengthSums(nBins, 0.0); // total length in each bin
    std::vector<cv::Vec3f>::iterator rtlIter;
    for(rtlIter = filteredLines.begin(); rtlIter != filteredLines.end(); ++rtlIter){
        int index = static_cast<int>( ( (*rtlIter)[1] + PI/2 ) / intervalBins );
        if(index >= 0 && index < nBins){
            bins[index].push_back(*rtlIter);
            binLengthSums[index] += (*rtlIter)[2];
        }
    }

    //=====================================================
    // 5 sort the total length of each bin

    std::vector<NumberIndex> numbersWithIndex;

    // 输出每个 bin 中的 length 总和
    for (int i = 0; i < nBins; ++i) {
        std::cout << "Bin " << i << " total length: " << binLengthSums[i] << std::endl;
    }

    // 将数组转化为结构体向量
    for (int i = 0; i < binLengthSums.size(); ++i) {
        numbersWithIndex.push_back({binLengthSums[i], i});
    }

    // 对结构体向量按照数字大小进行排序
    std::sort(numbersWithIndex.begin(), numbersWithIndex.end(),
              [](const NumberIndex& a, const NumberIndex& b) {
                  return a.number > b.number;
              });

    int index[2];
    index[0] = numbersWithIndex[0].index;
    index[1] = numbersWithIndex[1].index;
    
    //=====================================================
    // 6. get mean theta and rho of 2 bins
    // To do: use K means to get the principle cluster in the bins

    std::vector<cv::Vec2f> result;
    
    for(auto ind: index){
        // std::cout<<"size" <<ind<<std::endl;
        double sum = 0;
        for(const auto& vec: bins[ind]){
            sum += vec[0];
        }
        float meanRho = sum/static_cast<float>(bins[ind].size());
        sum = 0;
        for(const auto& vec: bins[ind]){
            sum += vec[1];
        }
        float meanTheta = sum/static_cast<float>(bins[ind].size());
        if(meanTheta < 0){
            meanRho = -meanRho;
        }
        cv::Vec2f res(meanRho, meanTheta);
        result.push_back(res);
    }

    //=====================================================
    // 7 detect bottom line
    // To do use K means to find the bottom line


    return result;
}