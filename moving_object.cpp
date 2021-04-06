#include "moving_object.hpp"

#include <random>

static inline int sqr(int const x) { return x * x; }

// static
MovingObject MovingObject::MakeCircle(int radius, std::array<uchar,3> const& color, Path const& p) {
    MovingObject object;
    object.shape_.reset(new Circle(radius, color));
    object.setPath(p);
    return object;
}

// static
MovingObject MovingObject::MakeRect(int w, int h, std::array<uchar,3> const& color, Path const& p) {
    MovingObject object;
    object.shape_.reset(new Rect(w, h, color));
    object.setPath(p);
    return object;
}

void MovingObject::setPath(Path const& p) {
    path = p;
    x = p.origin[0];
    y = p.origin[1];
}

void MovingObject::draw(cv::Mat& frame) {
    shape_->draw(frame, x, y);
}

void MovingObject::update() {
    x += path.step[0];
    y += path.step[1];
}

std::vector<MovingObject> generateRandomMovingCircles(int num, int const frame_width, int const frame_height)
{
    std::vector<MovingObject> objects;
    std::mt19937_64 rand_engine;
    std::uniform_real_distribution<float> rand_gen;
    for(int n = 0; n < num; ++n) {
        std::array<uchar,3> rand_color;
        rand_color[0] = uchar(255 * rand_gen(rand_engine));
        rand_color[1] = uchar(255 * rand_gen(rand_engine));
        rand_color[2] = uchar(255 * rand_gen(rand_engine));

        Path rand_path;
        rand_path.origin[0] = frame_width / 2 + int((frame_width * (rand_gen(rand_engine) - 0.5)));
        rand_path.origin[1] = frame_height / 2 + int((frame_height * (rand_gen(rand_engine) - 0.5)));
        rand_path.step[0] = int(4 * (rand_gen(rand_engine) - 0.5));
        rand_path.step[1] = int(4 * (rand_gen(rand_engine) - 0.5));

        int const radius = int(75 * rand_gen(rand_engine));
        objects.emplace_back(MovingObject::MakeCircle(radius, rand_color, rand_path));
    }
    return objects;
}

std::vector<MovingObject> generateRandomMovingRects(int num, int const frame_width, int const frame_height)
{
    std::vector<MovingObject> objects;
    std::mt19937_64 rand_engine;
    std::uniform_real_distribution<float> rand_gen;
    for(int n = 0; n < num; ++n) {
        std::array<uchar,3> rand_color;
        rand_color[0] = uchar(255 * rand_gen(rand_engine));
        rand_color[1] = uchar(255 * rand_gen(rand_engine));
        rand_color[2] = uchar(255 * rand_gen(rand_engine));

        Path rand_path;
        rand_path.origin[0] = frame_width / 2 + int((frame_width * (rand_gen(rand_engine) - 0.5)));
        rand_path.origin[1] = frame_height / 2 + int((frame_height * (rand_gen(rand_engine) - 0.5)));
        rand_path.step[0] = int(4 * (rand_gen(rand_engine) - 0.5));
        rand_path.step[1] = int(4 * (rand_gen(rand_engine) - 0.5));

        int const width = int(80 * rand_gen(rand_engine));
        int const height = int(80 * rand_gen(rand_engine));
        objects.emplace_back(MovingObject::MakeRect(width, height, rand_color, rand_path));
    }
    return objects;
}