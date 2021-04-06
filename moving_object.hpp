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
    static MovingObject MakeCircle(int radius, std::array<uchar,3> const& color, Path const& p);
    static MovingObject MakeRect(int w, int h, std::array<uchar,3> const& color, Path const& p);
    
    void setPath(Path const& p);
    void draw(cv::Mat& frame);
    void update();

  private:
    std::unique_ptr<Shape> shape;
    Path path;
    int x, y;
};

std::vector<MovingObject> generateRandomMovingCircles(int num, int const frame_width, int const frame_height);
std::vector<MovingObject> generateRandomMovingRects(int num, int const frame_width, int const frame_height);

#endif // MANOJ_OPENCV_MOVING_OBJECT_HPP
