/**
 * @file feature_detection.cpp
 * @author Yang Jiaye (yjy420@sjtu.edu.cn)
 * @brief feature detection
 * @version 0.1
 * @date 2024-05-28
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "feature_detection.h"

typedef FeatureDetector::HoughCoord HoughCoord;
typedef FeatureDetector::Line Line;

double FeatureDetector::computeLength(Line line)
{
    double x1 = line[0];
    double y1 = line[1];
    double x2 = line[2];
    double y2 = line[3];

    return std::sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

cv::Vec2d FeatureDetector::computeIntersection(HoughCoord line1, HoughCoord line2) {
    double rho1 = line1[0];
    double theta1 = line1[1];
    double rho2 = line2[0];
    double theta2 = line2[1];

    double y_intersect = (rho1 * std::cos(theta2) - rho2 * std::cos(theta1)) / std::sin(theta1 - theta2);
    double x_intersect = (rho1 * std::sin(theta2) - rho2 * std::sin(theta1)) / std::sin(theta2 - theta1);

    return cv::Vec2f(x_intersect, y_intersect);
}

HoughCoord FeatureDetector::computeHoughLine(Line line)
{
    double x1 = line[0];
    double y1 = line[1];
    double x2 = line[2];
    double y2 = line[3];

    double rho, theta;

    if(y1 == y2){
        theta = PI/2;
    }
    double k_inv = (x1 - x2) / (y1 - y2);
    theta = std::atan(-k_inv);

    double A = y2 - y1;
    double B = x2 - x1;
    double C = x2*y1 - x1*y2;
    double length = std::sqrt(A*A + B*B);

    rho = abs(C) / length;

    double x0 = -k_inv * y1 + x1;
    if(x0 < 0){
        rho = -rho;
    }

    return HoughCoord(rho, theta);
}

void FeatureDetector::drawHoughLine(cv::Mat& image, HoughCoord line, cv::Scalar color, int thickness)
{
    double rho = line[0];
    double theta = line[1];

    double a = std::cos(theta);
    double b = std::sin(theta);
    double x0 = a * rho;
    double y0 = b * rho;

    cv::Point pt1(cvRound(x0 + 2000 * (-b)), cvRound(y0 + 2000 * (a)));
    cv::Point pt2(cvRound(x0 - 2000 * (-b)), cvRound(y0 - 2000 * (a)));

    cv::line(image, pt1, pt2, color, thickness);
}

HoughCoord FeatureDetector::fitLineFromLines(LineVec lineVec)
{
    std::vector<cv::Vec2d> points;
    std::vector<double> weights;

    // Iterate over each line in lineVec
    for (auto line : lineVec) {
        cv::Vec2d pt1(line[0], line[1]);
        cv::Vec2d pt2(line[2], line[3]);

        double length = cv::norm(pt2 - pt1);
        points.push_back(pt1);
        points.push_back(pt2);
        weights.push_back(length);
        weights.push_back(length);
    }

    // Use Least Square Regression to find the line
    HoughCoord result = fitLineFromPoints(points, weights);

    return result;
}

// Define the function fitLineFromPoints
HoughCoord FeatureDetector::fitLineFromPoints(const std::vector<cv::Vec2d>& points, std::vector<double> weights)
{
    // Constructing matrices A and b for the least squares problem Ax = b
    Eigen::MatrixXd A(points.size(), 2);
    Eigen::VectorXd b(points.size());
    
    // Fill matrices A and b with data from points
    for (int i = 0; i < points.size(); i++) {
        A(i, 0) = points[i][0];
        A(i, 1) = 1;
        b(i) = points[i][1];
    }

    // If weights is empty, set all weights to 1
    if(weights.size() == 0){
        weights = std::vector<double>(points.size(), 1);
    }
    if(weights.size() != points.size()){
        std::cerr<<"[fitLineFromPoints]:\tweights size not equal to points size\n";
        return cv::Vec2d(0, 0);
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

    // hough coord result
    HoughCoord result;
    if (m == 0) {
        result[1] = PI / 2; // Pi/2 radians = 90 degrees
        result[0] = c;
    } else {
        result[1] = std::atan(-1 / m);
        result[0] = c * std::sin(result[1]);
    }

    return result;
}

std::vector<HoughCoord> FeatureDetector::detectRunwayLine(cv::Mat image, bool if_save)
{
    //=====================================================
    // 1. preprocess image

    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    //=====================================================
    // 2. LSD

    // create LSD object
    cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector(cv::LSD_REFINE_STD);
    LineVec lsdLines; // storage line features

    // detect line feature
    double startTime = double(cv::getTickCount());

    // use opencv lsd implementation here
    lsd->detect(gray, lsdLines);

    double endTime = double(cv::getTickCount());
    double durationTime = (endTime - startTime) * 1000 / cv::getTickFrequency();
    std::cout << "[detectRunwayLine]:\tIt took " << durationTime << " ms to detect line features." << std::endl;

    if(if_save)
    {
        cv::Mat drawImage = gray;
        lsd->drawSegments(drawImage, lsdLines);
        cv::imwrite("./lsdImage.png", drawImage);
    }

    /*
    the most difficult part is to find specific 3 lines from 
    all the detected lines above. We filter out short line 
    below threshold, and rotation angle lower threshold first. 
    Find the 2 slope interval which has the largest 
    accumulating length. Use mean theta and d to find the 2 slant 
    runway lines. Between the 2 lines repeat the procedure above
    to find the bottom line.
    */
    
    //=====================================================
    // 3 filter out lines and group them

    // in filteredLines, each Vec4f storage coords of a line
    LineVec filteredLines;
    LineVec::iterator lineIter; // iterate throughout lines_std
    double rho, theta, length;
    double lengthThreshold = 10; //length threshold
    double angleThreshold = PI/4; //angle threshold
    for(lineIter = lsdLines.begin(); lineIter != lsdLines.end(); ++lineIter)
    {
        //rho theta
        HoughCoord houghCoord = computeHoughLine(*lineIter);
        theta = houghCoord[1];
        length = computeLength(*lineIter);
        if(
            length > lengthThreshold 
            && abs(theta) < angleThreshold
        ){
            filteredLines.push_back(*lineIter);
        }
    }

    if(if_save)
    {
        cv::Mat drawImage = gray;
        for(auto line: filteredLines){
            cv::line(drawImage, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(0, 255, 0), 2);
        }
        cv::imwrite("./filteredLines.png", drawImage);
    }

    //=====================================================
    // 4 use hough transform to detect runway

    std::vector<HoughCoord> results = filterLines(image, filteredLines, if_save);

    return results;
}

std::vector<cv::RotatedRect> FeatureDetector::detectThreshold(cv::Mat image, std::vector<HoughCoord> lines, bool if_save, bool if_debug)
{
    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Threshold the image to get binary image
    cv::Mat thresh;
    cv::threshold(gray, thresh, 220, 255, 0);
    if(if_save){
        cv::imwrite("thresh.png", thresh);
    }

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw blue contours on the original image
    if(if_save){
        cv::Mat contours_image = image.clone();
        cv::drawContours(contours_image, contours, -1, cv::Scalar(255, 0, 0), 2);
        cv::imwrite("contours.png", contours_image);
    }

    // Create an empty image for rotated rectangles and centroids
    cv::Mat bars_image = image.clone();
    cv::Mat centroids_image = image.clone();
    std::vector<cv::RotatedRect> bars;

    // Process each contour
    for (const auto& contour : contours) {
        
        // Get the rotated rectangle that best fits the contour
        cv::RotatedRect rect = cv::minAreaRect(contour);
        bars.push_back(rect);

        if(if_save){
            // Draw the rotated rectangle in white
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++){
                cv::line(bars_image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 255));
                cv::line(centroids_image, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 0, 255));
            }

            // Calculate the centroid of the rectangle
            cv::Point2f center = rect.center;

            // Draw the centroid in red
            cv::circle(centroids_image, center, 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    if(if_save){
        cv::imwrite("bars.png", bars_image);
        cv::imwrite("centroids.png", centroids_image);
    }
    if(if_debug){
        std::cout << "[detectThreshold]:\tfiltering bars\n";
    }
    
    bars = filterThresholdBars(bars, lines, if_save, image.clone(), if_debug);

    // draw bars
    if(if_save){
        cv::Mat filtered_bars_image = image.clone();
        for (const auto& thresh : bars) {
            cv::Point2f vertices[4];
            thresh.points(vertices);
            for (int i = 0; i < 4; i++){
                cv::line(filtered_bars_image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 255));
            }
        }
        cv::imwrite("filtered_bars.png", filtered_bars_image);
    }

    return bars;
}

// Function to detect the bottom line and the upper line
std::vector<HoughCoord> FeatureDetector::detectBottomUpperLines(const std::vector<cv::RotatedRect>& bars)
{
    std::vector<cv::Vec2d> lower_points;
    std::vector<cv::Vec2d> upper_points;

    for(auto bar : bars){
        cv::Point2f vertices[4];
        bar.points(vertices);
        // sort vertices by y
        std::sort(vertices, vertices+4, [](cv::Point2f a, cv::Point2f b){
            return a.y < b.y;
        });
        lower_points.push_back(cv::Vec2f(vertices[0].x, vertices[0].y));
        upper_points.push_back(cv::Vec2f(vertices[3].x, vertices[3].y));
    }

    // Use Least Square Regression to find the bottom line and the upper line
    auto bottomLine = fitLineFromPoints(lower_points);
    auto upperLine = fitLineFromPoints(upper_points);

    return {bottomLine, upperLine};
}

// Function to detect the slope line
HoughCoord FeatureDetector::detectSlopeLine(std::vector<HoughCoord>BULines, std::vector<HoughCoord> LRLines)
{
    // Find intersection of bottom line and left line

    cv::Vec2d left_intersect = computeIntersection(BULines[0], LRLines[1]);
    cv::Vec2d right_intersect = computeIntersection(BULines[1], LRLines[0]);

    HoughCoord result = computeHoughLine(Line(left_intersect[0], left_intersect[1], right_intersect[0], right_intersect[1]));

    return result;
}

std::vector<HoughCoord> FeatureDetector::filterLines(cv::Mat bgrImage, std::vector<Line> lines, bool if_save)
{
    // number of steps in rho and theta range
    int NRho = 50;
    int NTheta = 50;
    double maxRho = sqrt(bgrImage.cols * bgrImage.cols + bgrImage.rows * bgrImage.rows);

    // Define evenly spaced values from -pi/2 to pi/2 for theta bins
    std::vector<double> theta_bins(NTheta + 1);
    for (int i = 0; i < NTheta + 1; ++i) {
        theta_bins[i] = i * PI / NTheta - PI / 2;
    }
    // Define evenly spaced values from -max to max for rho bins
    std::vector<double> rho_bins(NRho + 1);
    for (int i = 0; i < NRho + 1; ++i) {
        rho_bins[i] = i * 2 * maxRho / NRho - maxRho;
    }

    Eigen::MatrixXf houghAccumulater = Eigen::MatrixXf::Zero(NRho, NTheta);
    LineVecGrid lineVecGrid(NRho, std::vector<LineVec>(NTheta, LineVec(0)));

    // map rho and theta to grid
    for (auto line:lines)
    {
        //convert coord
        HoughCoord houghCoord = computeHoughLine(line);
        double rho = houghCoord[0];
        double theta = houghCoord[1];
        double length = computeLength(line);

        //find corresbonding grid index
        int rho_idx = std::distance(rho_bins.begin(), std::upper_bound(rho_bins.begin(), rho_bins.end(), rho)) - 1;
        int theta_idx = std::distance(theta_bins.begin(), std::upper_bound(theta_bins.begin(), theta_bins.end(), theta)) - 1;

        // boundary check
        if(theta_idx > NTheta-1){
            theta_idx = NTheta-1;
        }
        if(theta_idx < 0){
            theta_idx = 0;
        }
        if(rho_idx > NRho-1){
            rho_idx = NRho-1;
        }
        if(rho_idx < 0){
            rho_idx = 0;
        }

        // accumulator increase
        houghAccumulater(rho_idx, theta_idx) += length;
        // storage lines in LineVecGrid
        lineVecGrid[rho_idx][theta_idx].push_back(line);
    }

    // number of lines to preserve, sorted by total length in accumulator
    int line_num = 2;
    std::vector<HoughCoord> results;

    for(int i = 0; i<line_num; i++)
    {
        Eigen::Index max_row, max_col;
        // find max value
        double max_value = houghAccumulater.maxCoeff(&max_row, &max_col);
        houghAccumulater(max_row, max_col) = -INFINITY;
        LineVec lineVec = lineVecGrid[max_row][max_col];
        // use least square regression to fit the line
        results.push_back(fitLineFromLines(lineVec));
    }
    if(results[1][1] > results[0][1]){
        std::swap(results[0], results[1]);
    }

    return results;
}


// Filter out the rectangles by aspect ratios, rotated angles, position of center points and verticle scan
std::vector<cv::RotatedRect> FeatureDetector::filterThresholdBars(std::vector<cv::RotatedRect> bars, std::vector<HoughCoord> lines, bool if_save, cv::Mat runway_line_detect, bool if_debug)
{
    std::vector<cv::RotatedRect> filtered_bars1;
    std::vector<float> min_y;
    std::vector<float> max_y;
    std::vector<float> center_y;

    if(if_debug){
        std::cout<<"[filterThresholdBars]:     number of bars before filter: "<<bars.size()<<std::endl;
    }
        
    // check each threshold
    for (const auto& thresh : bars) {
        // Set flag to true
        bool is_thresh = true;

        float height = std::max(thresh.size.height, thresh.size.width);
        float width = std::min(thresh.size.height, thresh.size.width);

        // 1. Check if the aspect ratio of the rectangle is larger than 2
        float aspect_ratio = height / width;
        if (aspect_ratio < 2) {
            is_thresh = false;
        }
        if(if_debug){
            std::cout<<"aspect ratio filter ";
            std::cout<< is_thresh<<" "<<aspect_ratio<<" " <<height<<" "<<width<<std::endl;
        }
            
        // 2. Check if the angle is between 2 runway lines

        // compute distance between point 0,1 and that between point 1,2
        // and check direction of threshold
        cv::Point2f vertices[4];
        thresh.points(vertices);
        float dist1 = cv::norm(vertices[0] - vertices[1]);
        float dist2 = cv::norm(vertices[1] - vertices[2]);
        float angle;
        if (dist1 > dist2) {
            angle = std::atan2(vertices[0].y - vertices[1].y, vertices[0].x - vertices[1].x);
        } else {
            angle = std::atan2(vertices[1].y - vertices[2].y, vertices[1].x - vertices[2].x);
        }
        angle = angle * 180 / PI;
        if(angle < 0){
            angle += 180;
        }
        angle = angle - 90;// hough transform theta is 90 - angle

        float angle_L = lines[0][1] * 180 / PI;
        float angle_R = lines[1][1] * 180 / PI;
        float min_angle = angle_R - 0.5 * (angle_L - angle_R);
        float max_angle = angle_L + 0.5 * (angle_L - angle_R);

        if (angle < min_angle || angle > max_angle) {
            is_thresh = false;
        }

        if(if_debug){
            std::cout<<"angle filter ";
            std::cout<< is_thresh<<" "<<min_angle<<" "<<angle<<" " <<max_angle<<std::endl;
        }
            

        // 3. check the center of the rectangle inside 2 lines
        cv::Point2f center = thresh.center;
        float rho_L = lines[0][0];
        float theta_L = lines[0][1];
        float rho_R = lines[1][0];
        float theta_R = lines[1][1];
        // rho = x * cos(theta) + y * sin(theta)
        // compute intersect of 2 lines and y = center.y
        float x_L = (rho_L - center.y * std::sin(theta_L)) / std::cos(theta_L);
        float x_R = (rho_R - center.y * std::sin(theta_R)) / std::cos(theta_R);
        if (center.x < x_L || center.x > x_R) {
            is_thresh = false;
        }

        if(if_debug){
            std::cout<<"center filter ";
            std::cout<<is_thresh<<" "<<x_L<<" "<<center.x<<" "<<x_R<<std::endl;
            std::cout<<rho_L<<" "<<theta_L<<" "<<rho_R<<" "<<theta_R<<std::endl;
        }
           
        // If the rectangle is parallel to the runway lines, keep it
        if (is_thresh) {
            filtered_bars1.push_back(thresh);
            
            std::vector<float> y_coords;
            for (const auto& vertex : vertices) {
                y_coords.push_back(vertex.y);
            }

            auto iter = std::max_element(y_coords.begin(), y_coords.end());
            max_y.push_back(*iter);
            iter = std::min_element(y_coords.begin(), y_coords.end());
            min_y.push_back(*iter);
            center_y.push_back(center.y);
        }
    }
    if(if_debug){
        std::cout<<"[filterThresholdBars]:\tumber of bars after filter: "<<filtered_bars1.size()<<std::endl;
    }
        
    if(if_save){
        for(auto thresh: filtered_bars1){
            cv::Point2f vertices[4];
            thresh.points(vertices);
            for(int i = 0; i < 4; i++){
                cv::line(runway_line_detect, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 255), 2);
            }
        }
        cv::imwrite("filtered_bars1.png", runway_line_detect);
    }

    // 4 check each y = center.y in center_y, find the number of rects that min_y < y < max_y, and find the center.y of max number
    if(if_debug){
        std::cout<<"[filterThresholdBars]:\tcenter filter\n";
    }
        
    int max_count = 0;
    float max_center_y = 0;
    for (int i = 0; i < center_y.size(); i++) {
        int count = 0;
        for (int j = 0; j < center_y.size(); j++) {
            if (center_y[i] > min_y[j] && center_y[i] < max_y[j]) {
                count++;
            }
        }
        if (count > max_count) {
            max_count = count;
            max_center_y = center_y[i];
            // std::cout<<"=================center filter================\n";
            // std::cout<<max_count<<" "<<max_center_y<<std::endl;
        }
    }

    std::vector<cv::RotatedRect> filtered_bars2;

    for (int i = 0; i < center_y.size(); i++) {
        // std::cout<<"=================center filter================\n";
        // std::cout<<min_y[i]<<" "<<max_center_y<<" "<<max_y[i]<<std::endl;
        if (min_y[i] < max_center_y && max_center_y < max_y[i]) {
            filtered_bars2.push_back(filtered_bars1[i]);
        }
    }

    // 5. check the area of bars
    
    if(if_debug){
        std::cout<<"[filterThresholdBars]:\tarea filter\n";
    }
        

    std::vector<float> areas;
    // std::cout<<filtered_bars2.size()<<std::endl;
    if(filtered_bars2.size() == 0){
        std::cout<<"[filterThresholdBars]:\tno bars\n";
        return filtered_bars2;
    }

    for (const auto& thresh : filtered_bars2) {
        areas.push_back(thresh.size.area());
    }
    // median area
    std::sort(areas.begin(), areas.end());
    float median_area = areas[areas.size() / 2];
    for(auto iter = filtered_bars2.begin(); iter != filtered_bars2.end();){
        if(iter->size.area() < median_area*0.7 || iter->size.area() > median_area*1.4){
            // std::cout<<"=================area filter================\n";
            // std::cout<<median_area<<" "<<iter->size.area() <<std::endl;
            iter = filtered_bars2.erase(iter);
        }else{
            ++iter;
        }
    }

    // std::cout<<"=================return bars================\n";
    return filtered_bars2;
}