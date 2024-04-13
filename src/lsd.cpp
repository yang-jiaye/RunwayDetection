#include "lsd.h"

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

cv::Vec2f computeResults(LineVec lineVec){
    cv::Vec2f result;
    // std::cout<<"=================begin compute results==============\n";

    std::vector<cv::Vec2f> points;
    std::vector<double> weights;

    // Iterate over each line in lineVec
    for (auto line : lineVec) {
        cv::Vec2f pt1(line[0], line[1]);
        cv::Vec2f pt2(line[2], line[3]);

        double length = cv::norm(pt2 - pt1);
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

cv::Vec2f computeIntersection(cv::Vec2f line1, cv::Vec2f line2) {
    double rho1 = line1[0];
    double theta1 = line1[1];
    double rho2 = line2[0];
    double theta2 = line2[1];

    double y_intersect = (rho1 * std::cos(theta2) - rho2 * std::cos(theta1)) / std::sin(theta1 - theta2);
    double x_intersect = (rho1 * std::sin(theta2) - rho2 * std::sin(theta1)) / std::sin(theta2 - theta1);

    return cv::Vec2f(x_intersect, y_intersect);
}

// Define the function LeastSquareRegression
cv::Vec2f LeastSquareRegression(const std::vector<cv::Vec2f>& points) {
    // Implementation of the function
    // Constructing matrices A and b for the least squares problem Ax = b
    Eigen::MatrixXd A(points.size(), 2);
    Eigen::VectorXd b(points.size());
    
    // Fill matrices A and b with data from points
    for (int i = 0; i < points.size(); i++) {
        A(i, 0) = points[i][0];
        A(i, 1) = 1;
        b(i) = points[i][1];
    }

    // Solve the least squares problem using Eigen
    Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
    cv::Vec2f result(x(0), x(1));

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

std::vector<cv::Vec2f> detectRunwayLine(cv::Mat bgrImage, bool writeImage)
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

std::vector<cv::RotatedRect> detectRunwayThreshold(cv::Mat bgrImage, std::vector<cv::Vec2f> lines, bool writeImage)
{
    // Convert the image to grayscale
    cv::Mat gray;
    cv::cvtColor(bgrImage, gray, cv::COLOR_BGR2GRAY);

    // Threshold the image to get binary image
    cv::Mat thresh;
    cv::threshold(gray, thresh, 220, 255, 0);
    if(writeImage){
        cv::imwrite("thresh.png", thresh);
    }

    // Find contours in the binary image
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Draw blue contours on the original image
    if(writeImage){
        cv::Mat contours_image = bgrImage.clone();
        cv::drawContours(contours_image, contours, -1, cv::Scalar(255, 0, 0), 2);
        cv::imwrite("contours.png", contours_image);
    }

    // Create an empty image for rotated rectangles and centroids
    cv::Mat threshs_image = bgrImage.clone();
    cv::Mat centroids_image = bgrImage.clone();
    std::vector<cv::RotatedRect> threshs;

    // Process each contour
    for (const auto& contour : contours) {
        
        // Get the rotated rectangle that best fits the contour
        cv::RotatedRect rect = cv::minAreaRect(contour);
        threshs.push_back(rect);

        if(writeImage){
            // Draw the rotated rectangle in white
            cv::Point2f vertices[4];
            rect.points(vertices);
            for (int i = 0; i < 4; i++){
                cv::line(threshs_image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 255));
                cv::line(centroids_image, vertices[i], vertices[(i+1)%4], cv::Scalar(255, 255, 255));
            }

            // Calculate the centroid of the rectangle
            cv::Point2f center = rect.center;

            // Draw the centroid in red
            cv::circle(centroids_image, center, 1, cv::Scalar(0, 0, 255), -1);
        }
    }

    if(writeImage){
        cv::imwrite("threshs.png", threshs_image);
        cv::imwrite("centroids.png", centroids_image);
    }

    threshs = filterThresholds(threshs, lines);

    // draw threshs
    if(writeImage){
        cv::Mat filtered_threshs_image = bgrImage.clone();
        for (const auto& thresh : threshs) {
            cv::Point2f vertices[4];
            thresh.points(vertices);
            for (int i = 0; i < 4; i++){
                cv::line(filtered_threshs_image, vertices[i], vertices[(i+1)%4], cv::Scalar(0, 255, 255));
            }
        }
        cv::imwrite("filtered_threshs.png", filtered_threshs_image);
    }

    return threshs;
}

// Filter out the rectangles by aspect ratios, rotated angles, position of center points and verticle scan
std::vector<cv::RotatedRect> filterThresholds(std::vector<cv::RotatedRect> threshs, std::vector<cv::Vec2f> lines)
{
    std::vector<cv::RotatedRect> filtered_threshs;
    std::vector<float> min_y;
    std::vector<float> max_y;
    std::vector<float> center_y;

    for (const auto& thresh : threshs) {
        // Set flag to true
        bool is_thresh = true;

        float height = std::max(thresh.size.height, thresh.size.width);
        float width = std::min(thresh.size.height, thresh.size.width);

        // 1. Check if the aspect ratio of the rectangle is larger than 2
        float aspect_ratio = height / width;
        if (aspect_ratio < 2) {
            is_thresh = false;
        }

        // 2. Check if the angle is close to the angle of any of the runway lines

        // check distance between point 0,1 and that between point 1,2
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
            // std::cout<<"=================angle filter================\n";
            // std::cout<<angle<<" "<<min_angle<<" "<<max_angle<<std::endl;
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
            // std::cout<<"=================center filter================\n";
            // std::cout<<center.x<<" "<<x_L<<" "<<x_R<<std::endl;
        }

        // If the rectangle is parallel to the runway lines, keep it
        if (is_thresh) {
            filtered_threshs.push_back(thresh);
            
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

    // 4 check each y = center.y in center_y, find the number of rects that min_y < y < max_y, and find the center.y of max number
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

    std::vector<cv::RotatedRect> filtered_threshs2;

    for (int i = 0; i < center_y.size(); i++) {
        // std::cout<<"=================center filter================\n";
        // std::cout<<min_y[i]<<" "<<max_center_y<<" "<<max_y[i]<<std::endl;
        if (min_y[i] < max_center_y && max_center_y < max_y[i]) {
            filtered_threshs2.push_back(filtered_threshs[i]);
        }
    }

    // 5. check the area of threshs
    std::vector<float> areas;
    for (const auto& thresh : filtered_threshs2) {
        areas.push_back(thresh.size.area());
    }
    // median area
    std::sort(areas.begin(), areas.end());
    float median_area = areas[areas.size() / 2];
    for(auto iter = filtered_threshs2.begin(); iter != filtered_threshs2.end();){
        if(iter->size.area() < median_area/2 || iter->size.area() > median_area*2){
            // std::cout<<"=================area filter================\n";
            // std::cout<<median_area<<" "<<iter->size.area() <<std::endl;
            iter = filtered_threshs2.erase(iter);
        }else{
            ++iter;
        }
    }

    return filtered_threshs2;
}

// Function to detect the bottom line and the upper line
std::vector<cv::Vec2f> detectBottomAndUpperLines(const std::vector<cv::RotatedRect>& threshs)
{
    std::vector<cv::Vec2f> lower_points;
    std::vector<cv::Vec2f> upper_points;

    for(auto thres : threshs){
        cv::Point2f vertices[4];
        thres.points(vertices);
        // sort vertices by y
        std::sort(vertices, vertices+4, [](cv::Point2f a, cv::Point2f b){
            return a.y < b.y;
        });
        lower_points.push_back(cv::Vec2f(vertices[0].x, vertices[0].y));
        upper_points.push_back(cv::Vec2f(vertices[3].x, vertices[3].y));
    }

    // Use Least Square Regression to find the bottom line and the upper line
    auto bottomLine = LeastSquareRegression(lower_points);
    auto upperLine = LeastSquareRegression(upper_points);

    return {bottomLine, upperLine};
}

// Function to detect the slope line
std::vector<cv::Vec2f> detectSlopeLine(std::vector<cv::Vec2f>BULines, std::vector<cv::Vec2f> LRLines)
{
    // Find intersection of bottom line and left line

    cv::Vec2f left_intersect = computeIntersection(BULines[0], LRLines[1]);
    cv::Vec2f right_intersect = computeIntersection(BULines[1], LRLines[0]);

    return {left_intersect, right_intersect};
}