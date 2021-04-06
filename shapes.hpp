#ifndef MANOJ_OPENCV_SHAPES_HPP
#define MANOJ_OPENCV_SHAPES_HPP

#include "opencv2/core.hpp"

struct Shape
{
    virtual ~Shape() {}
    virtual void draw(cv::Mat& frame, int const x, int const y) = 0;
};

struct Circle : public Shape
{
    int radius;
    std::array<uchar,3> color;

    Circle(int r, std::array<uchar,3> c)
    : Shape(), radius(r), color(c)
    {}

    void draw(cv::Mat& frame, int const x0, int const y0) override {
        cv::Scalar const color_scalar(color[0], color[1], color[2]);
        cv::circle(frame, cv::Point{x0,y0}, radius, color_scalar, cv::FILLED);
    }
};

struct Rect : public Shape
{
    int width;
    int height;
    std::array<uchar,3> color;

    Rect(int w, int h, std::array<uchar,3> c)
    : width(w), height(h), color(c)
    {}

    void draw(cv::Mat& mat, int const x0, int const y0) override {
        cv::Point const top_left = cv::Point{x0,y0} - cv::Point{width/2, height/2};
        cv::Point const bottom_right = top_left + cv::Point{width, height};
        cv::Scalar const color_scalar(color[0], color[1], color[2]);
        cv::rectangle(mat, top_left, bottom_right, color_scalar, cv::FILLED);
    }
};

#endif // MANOJ_OPENCV_SHAPES_HPP
