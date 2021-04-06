
#include <algorithm>
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

#include "shapes.hpp"
#include "moving_object.hpp"


int constexpr FRAME_WIDTH = 960;
int constexpr FRAME_HEIGHT = 600;

inline int sqr(int const x) { return x * x; }

void generateFrames(std::vector<MovingObject>& circles,
                    std::vector<MovingObject>& rects,
                    std::string const& prefix)
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
        for(auto &obj : circles) {
            obj.draw(frame);
            obj.update();
        }
        for(auto &obj : rects) {
            obj.draw(frame);
            obj.update();
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
    std::vector<MovingObject> circles = generateRandomMovingCircles(100, FRAME_WIDTH, FRAME_HEIGHT);
    std::vector<MovingObject> rects = generateRandomMovingRects(100, FRAME_WIDTH, FRAME_HEIGHT);
    generateFrames(circles, rects, argv[1]);
    return 0;
}
