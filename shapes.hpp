#ifndef MANOJ_OPENCV_SHAPES_HPP
#define MANOJ_OPENCV_SHAPES_HPP

#include <array>

#include "opencv2/core.hpp"

using RgbColor = std::array<uchar,3>;
class Shape
{
  public:
    explicit Shape(RgbColor const& rgbColor);
    virtual ~Shape() {}
    virtual void draw(cv::Mat& frame, int const x, int const y) const = 0;
    virtual cv::Rect2i boundingBox(int const x, int const y) const = 0;
    RgbColor const& color() const;

  private:
    RgbColor color_;

};

struct Circle : public Shape
{
  public:

    Circle(int r, RgbColor const& c)
    : Shape(c), radius_(r)
    {}

    void draw(cv::Mat& frame, int const x0, int const y0) const override;
    cv::Rect2i boundingBox(int const x, int const y) const override;

  private:
    int radius_;
};

struct Rect : public Shape
{
  public:
    Rect(int w, int h, RgbColor const& c)
    : Shape(c), width_(w), height_(h)
    {}

    void draw(cv::Mat& mat, int const x0, int const y0) const override;
    cv::Rect2i boundingBox(int const x, int const y) const override;

  private:
    int width_;
    int height_;
};

inline Shape::Shape(RgbColor const& rgb_color)
: color_(rgb_color)
{    
}

inline RgbColor const& Shape::color() const {
    return this->color_;
}

inline cv::Rect2i Circle::boundingBox(int const x, int const y) const /*override*/ {
    return cv::Rect2i(x - this->radius_ -1, y - this->radius_ -1, 2 * this->radius_ + 2, 2 * this->radius_ + 2);
}

inline cv::Rect2i Rect::boundingBox(int const x, int const y) const /*override*/ {
    return cv::Rect2i(x - width_/2 -1, y - height_/2 -1, 2 * this->width_ + 2, 2 * this->height_ + 2);
}


#endif // MANOJ_OPENCV_SHAPES_HPP
