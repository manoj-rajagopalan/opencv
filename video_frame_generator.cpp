#include <array>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/videoio.hpp"

inline
uint32_t cvColorScalar(std::array<uchar,3> const rgb) // return OpenCV BGR order
{
    return uint32_t(rgb[0]) | (uint32_t(rgb[1]) << 8) | (uint32_t(rgb[2]) << 16);
}

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
        uint32_t const color_scalar = cvColorScalar(color);
        cv::rectangle(mat, top_left, bottom_right, color_scalar, cv::FILLED);
    }
};

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

    void update(cv::Mat& frame) {
        shape->draw(frame, x, y);
        x += path.step[0];
        y += path.step[1];
    }

private:
    std::unique_ptr<Shape> shape;
    Path path;
    int x, y;
};

int constexpr FRAME_WIDTH = 960;
int constexpr FRAME_HEIGHT = 600;

inline int sqr(int const x) { return x * x; }

std::vector<MovingObject> generateObjects(int num)
{
    std::vector<MovingObject> objects;
    std::mt19937_64 rand_engine;
    std::uniform_real_distribution<float> rand_gen;
    for(int n = 0; n < num; ++n) {
        float const coin_toss = rand_gen(rand_engine);
        std::array<uchar,3> rand_color;
        rand_color[0] = uchar(255 * rand_gen(rand_engine));
        rand_color[1] = uchar(255 * rand_gen(rand_engine));
        rand_color[2] = uchar(255 * rand_gen(rand_engine));

        Path rand_path;
        rand_path.origin[0] = FRAME_WIDTH / 2 + int((FRAME_WIDTH * (rand_gen(rand_engine) - 0.5)));
        rand_path.origin[1] = FRAME_HEIGHT / 2 + int((FRAME_HEIGHT * (rand_gen(rand_engine) - 0.5)));
        rand_path.step[0] = int(4 * (rand_gen(rand_engine) - 0.5));
        rand_path.step[1] = int(4 * (rand_gen(rand_engine) - 0.5));

        if(coin_toss < 0.5) {
            int const radius = int(75 * rand_gen(rand_engine));
            objects.emplace_back(MovingObject::MakeCircle(radius, rand_color, rand_path));

        } else {
            int const width = int(80 * rand_gen(rand_engine));
            int const height = int(80 * rand_gen(rand_engine));
            objects.emplace_back(MovingObject::MakeRect(width, height, rand_color, rand_path));
        }
    }
    return objects;
}

void generateFrames(std::vector<MovingObject>& objects, std::string const& prefix)
{
    // std::vector<MovingObject> objects;
    // objects.emplace_back(MovingObject::MakeCircle(50, {255u, 255u, 0u},
    //                                               Path{/*origin*/ {100,50}, /*step*/ {2,1}})
    //                     );
    // objects.emplace_back(MovingObject::MakeRect(50, 50, {0u, 255u, 255u},
    //                                             Path{/*origin*/ {500,50}, /*step*/ {-2,1}}));
    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
    cv::Mat rgb_frame(frame.size[0], frame.size[1], frame.type());

    for(int i_frame = 0; i_frame < 500; ++i_frame) {
        std::cout << "Drawing frame " << i_frame << '\n';
        frame = 0; // clear
        for(auto &obj : objects) {
            obj.update(frame);
        }
        // video_writer.write(frame);
        // cv::imshow("frame", frame);
        std::ostringstream s;
        s << prefix << '-' << std::setw(3) << std::setfill('0') << i_frame << ".png";
        
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        cv::imwrite(s.str(), rgb_frame);
        // cv::waitKey(10);
    }
}

int main(int const argc, char const *argv[])
{
    assert(argc == 2 && "Usage: video_frame_generator <prefix>-nnn.png");
    std::vector<MovingObject> objects = generateObjects(100);
    generateFrames(objects, argv[1]);
    return 0;
}
