#ifndef MANOJ_OPENCV_MOVING_OBJECT_HPP
#define MANOJ_OPENCV_MOVING_OBJECT_HPP

#include <array>
#include <memory>

#include "opencv2/core.hpp"

#include "shapes.hpp"

struct Path
{
    int origin[2];
    int step[2];
};

class MovingObject
{
  public:
    static MovingObject MakeCircle(int radius, std::array<uchar,3> const& color, Path const& p) {
        MovingObject object;
        object.shape.reset(new Circle(radius, color));
        object.setPath(p);
        return object;
    }

    static MovingObject MakeRect(int w, int h, std::array<uchar,3> const& color, Path const& p) {
        MovingObject object;
        object.shape.reset(new Rect(w, h, color));
        object.setPath(p);
        return object;
    }
    
    void setPath(Path const& p) {
        path = p;
        x = p.origin[0];
        y = p.origin[1];
    }

    void draw(cv::Mat& frame) {
        shape->draw(frame, x, y);
    }

    void update() {
        x += path.step[0];
        y += path.step[1];
    }

private:
    std::unique_ptr<Shape> shape;
    Path path;
    int x, y;
};

#endif // MANOJ_OPENCV_MOVING_OBJECT_HPP
