#include "shapes.hpp"

#include "opencv2/imgproc.hpp"

void Circle::draw(cv::Mat& frame, int const x0, int const y0) const /*override*/ {
    cv::Scalar const color_scalar(this->color()[0], this->color()[1], this->color()[2]);
    cv::circle(frame, cv::Point{x0,y0}, this->radius_, color_scalar, cv::FILLED);
}

void Rect::draw(cv::Mat& mat, int const x0, int const y0) const /*override*/ {
    cv::Point const top_left = cv::Point{x0,y0} - cv::Point{this->width_/2, this->height_/2};
    cv::Point const bottom_right = top_left + cv::Point{this->width_, this->height_};
    cv::Scalar const color_scalar(this->color()[0], this->color()[1], this->color()[2]);
    cv::rectangle(mat, top_left, bottom_right, color_scalar, cv::FILLED);
}
