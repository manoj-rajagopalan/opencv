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

    Circle(int r, std::array<uchar,3>const& c)
    : Shape(), radius(r), color(c)
    {}

    void draw(cv::Mat& frame, int const x0, int const y0) override;
};

struct Rect : public Shape
{
    int width;
    int height;
    std::array<uchar,3> color;

    Rect(int w, int h, std::array<uchar,3> c)
    : width(w), height(h), color(c)
    {}

    void draw(cv::Mat& mat, int const x0, int const y0) override;
};

#endif // MANOJ_OPENCV_SHAPES_HPP
