#include "shapes.hpp"

#include "opencv2/imgproc.hpp"

void Circle::draw(cv::Mat& frame, int const x0, int const y0) /*override*/ {
    cv::Scalar const color_scalar(color[0], color[1], color[2]);
    cv::circle(frame, cv::Point{x0,y0}, radius, color_scalar, cv::FILLED);
}

void Rect::draw(cv::Mat& mat, int const x0, int const y0) /*override*/ {
    cv::Point const top_left = cv::Point{x0,y0} - cv::Point{width/2, height/2};
    cv::Point const bottom_right = top_left + cv::Point{width, height};
    cv::Scalar const color_scalar(color[0], color[1], color[2]);
    cv::rectangle(mat, top_left, bottom_right, color_scalar, cv::FILLED);
}
