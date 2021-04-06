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

#include "draw_shapes.cu"

int constexpr FRAME_WIDTH = 960;
int constexpr FRAME_HEIGHT = 600;

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << ": " << message << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

void generateFrames(std::vector<MovingObject> const& circles,
                    std::string const& prefix)
{
    cv::Mat frame(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC3);
    int const frame_num_bytes = FRAME_HEIGHT * FRAME_WIDTH * 3;

    cudaError_t cuda_status = cudaSuccess;

    // CUDA stream
    cudaStream_t cuda_stream;
    cuda_status = cudaStreamCreate(&cuda_stream);
    cudaCheckSuccess(cuda_status, "Error creating CUDA stream");

    // Allocate frame on device
    uchar3 *dev_frame = nullptr;
    cuda_status = cudaMalloc((void**) &dev_frame, frame_num_bytes);
    cudaCheckSuccess(cuda_status, "Error allocating frame in device");

    for(int i_frame = 0; i_frame < 500; ++i_frame) {
        std::cout << "Drawing frame " << i_frame << '\n';

        // Clear frame
        dim2 clear_block(16,16);
        dim2 clear_grid((FRAME_WIDTH+15)/16, (FRAME_HEIGHT+15)/16);
        clearFrame<<<clear_grid, clear_block, 0, cuda_stream>>>(dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
        cuda_status = cudaGetLastError();
        cudaCheckSuccess(cuda_status, "Error launching 'clear' kernel in frame " + std::to_string(i_frame));

        // Draw circles
        int circle_counter = 0;
        for(auto &obj : circles) {
            Circle const *const c = dynamic_cast<Circle const*>(obj.shape.get());
            assert(c);
            dim2 const circle_block(2 * c->radius + 10, 2 * c->radius + 10);
            uchar3 const color{c->color[0], c->color[1], c->color[2]};
            drawCircle<<<1, circle_block, 0, cuda_stream>>>(obj.x, obj.y, c->radius, color, dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
            cuda_status = cudaGetLastError();
            cudaCheckSuccess(cuda_status, "Error launching kernel for circle " + std::to_string(circle_counter) + " in frame " + std::to_string(i_frame));
            obj.update();
            ++circle_counter;
        }

        // Fetch frame from GPU and write to img file
        cuda_status = cudaMemcpy((void*) frame.data, (void*) dev_frame, frame_num_bytes, cudaMemcpyDeviceToHost);
        cudaCheckSuccess(cuda_status, "Error copying data out of GPU in frame " + std::to_string(i_frame));
        std::ostringstream s;
        s << prefix << '-' << std::setw(3) << std::setfill('0') << i_frame << ".png";
        cv::imwrite(s.str(), frame);
    }

    cudaFree((void*) dev_frame);
}

int main(int const argc, char const *argv[])
{
    assert(argc == 2 && "Usage: video_frame_generator <prefix>-nnn.png");
    std::vector<MovingObject> circles = generateCircles(50);
    generateFrames(circles, argv[1]);
    return 0;
}
