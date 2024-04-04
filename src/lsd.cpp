#include "lsd.h"

std::vector<cv::Vec2f> runwayLineDetector(cv::Mat bgrImage, bool writeImage)
{
    //=====================================================
    // 1. preprocess image

    cv::Mat gray;
    cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);

    //=====================================================
    // 2. LSD

    // create LSD object
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

    // detect line feature
    std::vector<cv::Vec4f> lsdLines; // storage line features
    double startTime = double(cv::getTickCount());

    // use opencv lsd implementation here
    lsd->detect(gray, lsdLines);

    double endTime = double(cv::getTickCount());
    double durationTime = (endTime - startTime) * 1000 / cv::getTickFrequency();
    std::cout << "==================It took " << durationTime << " ms to detect line features.==================\n";

    if(writeImage)
    {
        cv::Mat drawImage = gray;
        lsd->drawSegments(drawImage, lsdLines);
        cv::imwrite("./lsdImage.png", drawImage);
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
    double rho, theta, length;
    double lengthThreshold = 10; //length threshold
    double angleThreshold = PI/4; //angle threshold
    for(lineIter = lsdLines.begin(); lineIter != lsdLines.end(); ++lineIter)
    {
        //rho theta length
        cv::Vec3f rtl = cartesianToHough(*lineIter);
        theta = rtl[1];
        length = rtl[2];
        if(
            length > lengthThreshold 
            && abs(theta) > angleThreshold
        ){
            filteredLines.push_back(*lineIter);
        }
    }

    cv::Mat mask(bgrImage.rows, bgrImage.cols, CV_8UC1, cv::Scalar(0));
    for (auto line:filteredLines)
    {
        cv::line(mask, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255), 1, cv::LINE_AA);
    }
    if(writeImage)
    {
        cv::imwrite("./mask.png", mask);
    }

    //=====================================================
    // 4 use hough transform to detect runway

    // std::vector<cv::Vec2f> results = houghTransformM(bgrImage, mask, writeImage);
    std::vector<cv::Vec2f> results = houghTransform(bgrImage, filteredLines, writeImage);

    return results;
}

std::vector<cv::Vec2f> houghTransform(cv::Mat bgrImage, std::vector<cv::Vec4f> lines, bool writeImage)
{
    int NRho = 50;
    int NTheta = 50;
    double maxRho = sqrt(bgrImage.cols * bgrImage.cols + bgrImage.rows * bgrImage.rows);
    double maxTheta = PI/2;

    // Define evenly spaced values from 0 to max for theta bins
    std::vector<double> theta_bins(NTheta);
    for (int i = 0; i < NTheta; ++i) {
        theta_bins[i] = i * PI / (NTheta - 1) - PI / 2;
    }
    // Define evenly spaced values from 0 to max for rho bins
    std::vector<double> rho_bins(NRho);
    for (int i = 0; i < NRho; ++i) {
        rho_bins[i] = i * maxRho / (NRho - 1);
    }

    Eigen::MatrixXf Accumulator = Eigen::MatrixXf::Zero(NRho-1, NTheta-1);
    LineGrid lineGrid(NRho-1, std::vector<LineVec>(NTheta-1, LineVec(0)));

    // 映射thetaList和rhoList到网格中
    for (auto line:lines)
    {
        //convert coord
        HoughCoord houghCoord = cartesianToHough(line);
        // std::cout<<houghCoord<<std::endl;
        //find corresbonding grid
        int rho_idx = std::distance(rho_bins.begin(), std::upper_bound(rho_bins.begin(), rho_bins.end(), houghCoord[0])) - 1;
        int theta_idx = std::distance(theta_bins.begin(), std::upper_bound(theta_bins.begin(), theta_bins.end(), houghCoord[1])) - 1;
        if(theta_idx >= NTheta-1){
            theta_idx = NTheta-2;
        }
        if(theta_idx < 0){
            theta_idx = 0;
        }
        // std::cout<<rho_idx<<" "<<theta_idx<<std::endl;
        //accumulator increase
        Accumulator(rho_idx, theta_idx) += houghCoord[2];
        //storage houghCoord
        lineGrid[rho_idx][theta_idx].push_back(line);
    }

    int line_num = 2;
    std::vector<cv::Vec2f> results;

    for(int i = 0; i<line_num; i++)
    {
        Eigen::Index max_row, max_col;
        // std::cout<<"=================find max value================\n";
        double max_value = Accumulator.maxCoeff(&max_row, &max_col);
        Accumulator(max_row, max_col) = -INFINITY;
        LineVec lineVec = lineGrid[max_row][max_col];
        results.push_back(computeResults(lineVec));
    }

    return results;
}

cv::Vec2f computeResults(LineVec lineVec){
    cv::Vec2f result;
    // std::cout<<"=================begin compute results==============\n";

    std::vector<Eigen::Vector2d> points;
    std::vector<double> weights;

    // Iterate over each line in lineVec
    for (auto line : lineVec) {
        Eigen::Vector2d pt1(line[0], line[1]);
        Eigen::Vector2d pt2(line[2], line[3]);

        double length = (pt2 - pt1).norm();
        points.push_back(pt1);
        points.push_back(pt2);
        weights.push_back(length);
        weights.push_back(length);
    }

    // Constructing matrices A and b for the least squares problem Ax = b
    Eigen::MatrixXd A(points.size(), 2);
    Eigen::VectorXd b(points.size());

    for (int i = 0; i < points.size(); ++i) {
        A(i, 0) = points[i][0];
        A(i, 1) = 1; // Corresponding to the intercept term
        b(i) = points[i][1];
    }

    // Diagonal matrix W for weighting
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(points.size(), points.size());
    for (int i = 0; i < weights.size(); ++i) {
        W(i, i) = weights[i];
    }

    // Solve the least squares problem using Eigen
    Eigen::VectorXd x = (A.transpose() * W * A).ldlt().solve(A.transpose() * W * b);

    double m = x(0); // Slope
    double c = x(1); // Intercept

    if (m == 0) {
        result[1] = M_PI / 2; // Pi/2 radians = 90 degrees
        result[0] = c;
    } else {
        result[1] = std::atan(-1 / m);
        result[0] = c * std::sin(result[1]);
    }

    return result;
}

std::vector<cv::Vec2f> houghTransformM(cv::Mat bgrImage, cv::Mat mask, bool writeImage)
{
    float rho_step = 10;
    float theta_step = PI/180 * 10;
    std::vector<cv::Vec2f> houghLines;
    HoughLines(mask, houghLines, rho_step, theta_step, 10);
    if(writeImage){
        cv::Mat houghImage = bgrImage.clone();
        for (auto line = houghLines.begin(); line != houghLines.begin() + 10; ++line) {
            float rho = (*line)[0];
            float theta = (*line)[1];
            std::cout<<rho<<" "<<theta<<std::endl;
            float x0 = rho * std::cos(theta);
            float y0 = rho * std::sin(theta);
            float x1 = static_cast<int>(x0 + 2000 * (std::sin(theta)));
            float y1 = static_cast<int>(y0 - 2000 * (std::cos(theta)));
            float x2 = static_cast<int>(x0 - 2000 * (std::sin(theta)));
            float y2 = static_cast<int>(y0 + 2000 * (std::cos(theta)));
            cv::line(houghImage, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("houghImage.png", houghImage);
    }

    int line_num = 2;
    houghLines.resize(line_num);
    // 对 lines 进行排序，按照直线的第二个元素（即 theta）降序排序
    std::sort(houghLines.begin(), houghLines.end(), [](const cv::Vec2f& a, const cv::Vec2f& b) {
        return a[1] > b[1];
    });

    // 选择前两条直线
    cv::Vec2f line_L = *houghLines.begin();
    cv::Vec2f line_R = *houghLines.end();
    std::vector<cv::Vec2f> results;
    results.push_back(line_L);
    results.push_back(line_R);

    //=====================================================
    // 7 detect bottom line
    // To do use K means to find the bottom line

    return results;
}

HoughCoord cartesianToHough(cv::Vec4f line)
{
    float x1 = line[0];
    float y1 = line[1];
    float x2 = line[2];
    float y2 = line[3];
    double theta = 0;
    if(x1 == x2){
        theta = PI/2;
    }
    double k = (y1 - y2) / (x1 - x2);
    theta = std::atan(k);
    double A = y2 - y1;
    double B = x2 - x1;
    double C = x2*y1 - x1*y2;
    double length = sqrt(A*A + B*B);
    double rho = abs(C) / length;
    cv::Vec3f rhoThetaLength(rho, theta, length);
    return rhoThetaLength;
}