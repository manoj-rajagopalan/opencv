/// Processes input video file and writes to output

#include <cassert>
#include <chrono>
#include <iostream>
#include <sstream>

#include "opencv2/videoio.hpp"

#include "video_properties.hpp"

int main(int const argc, char const *const argv[])
{
    assert(argc == 3 && "Usage: video_copy_frame_by_frame <infile> <outfile>");

    cv::VideoCapture video_infile;
    bool status = video_infile.open(argv[1]);
    assert(status && "Unable to open input file");

    VideoProperties const video_properties(video_infile);
    std::cout << "Input mat type = " << video_properties.cvMatType() << std::endl;
    // assert(video_properties.cv_mat_type == CV_8UC3);

    cv::VideoWriter video_outfile;
    status = video_outfile.open(argv[2], cv::VideoWriter::fourcc('a','v','c','1'),
                                video_properties.fps(), video_properties.frameSize(), video_properties.isColor());


    // CPU image frames
    cv::Mat frame;

    // Process frames
    for(int i_frame = 0; i_frame < video_properties.numFrames(); ++i_frame) {
        std::cout << "Frame " << i_frame << " ... ";

        auto /*timepoint*/ t_start = std::chrono::steady_clock::now();
        status = video_infile.read(frame);
        assert(status && "Unable to read input frame from video");
        assert(frame.isContinuous());
        
        video_outfile.write(frame);

        std::cout << " "
                  << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t_start).count()
                  << " us" << '\n';
    }

    return 0;
}
