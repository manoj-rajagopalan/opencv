#include <cassert>
#include <chrono>
#include <iostream>
#include <string>

#include <opencv2/videoio.hpp>

void printVideoFileProperties(const cv::VideoCapture& video_file_reader)
{
    using std::cout;
    using std::endl;

    cout << "- Number of frames = " << video_file_reader.get(cv::CAP_PROP_FRAME_COUNT) << endl;
    cout << "- Frame rate = " << video_file_reader.get(cv::CAP_PROP_FPS) << " fps" << endl;
    cout << "- Frame dimensions = " << video_file_reader.get(cv::CAP_PROP_FRAME_WIDTH) << 'x'
         << video_file_reader.get(cv::CAP_PROP_FRAME_HEIGHT) << endl;
    double fourcc_as_double = video_file_reader.get(cv::CAP_PROP_FOURCC);
    std::string fourcc((char*) &fourcc_as_double, sizeof(double));
    cout << "- FourCC = " << fourcc << endl;
    cout << "- FourCC (int) = 0x" << std::hex << int64_t(fourcc_as_double) << endl;
}

int main(int argc, char *argv[])
{
    assert(argc == 2 && "Usage: video_file_access <video file>");
    cv::VideoCapture video_file_reader;
    bool status = video_file_reader.open(argv[1]);
    assert(status && "Unable to open video file");

    printVideoFileProperties(video_file_reader);

    cv::Mat image;

    std::chrono::steady_clock clk;
    std::chrono::steady_clock::time_point t_begin = clk.now();
    status = video_file_reader.read(image);
    assert(status && "Unable to read frame");
    std::chrono::steady_clock::duration const load_latency = clk.now() - t_begin;
    std::cout << "load_latency = " << std::chrono::duration_cast<std::chrono::microseconds>(load_latency).count() << " us" << std::endl;

    return 0;
}