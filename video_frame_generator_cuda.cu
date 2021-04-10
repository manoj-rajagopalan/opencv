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

#include "draw_shapes.cuh"

int constexpr FRAME_WIDTH = 960;
int constexpr FRAME_HEIGHT = 600;

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << ": " << message << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

struct CudaThreadConfig
{
    explicit CudaThreadConfig(cv::Size2i const& bounding_box_extents)
    : bb_size(bounding_box_extents)
    {
        int const w = bb_size.width;
        int const h = bb_size.height;
        int const block_width = 32;
        int const block_height = 32;
        int const grid_width = (w + 31) / 32;
        int const grid_height = (h + 31) / 32;

        block.x = block_width;
        block.y = block_height;
        block.z = 1;

        grid.x = grid_width;
        grid.y = grid_height;
        grid.z = 1;
    }

    float fragmentation() const {
        int const bb_area = bb_size.width * bb_size.height;
        int const grid_area = (grid.x * block.x) * (grid.y * block.y);
        float const result = 1.0f - float(bb_area) / float(grid_area);
        return result;
    }

    cv::Size2i bb_size;
    dim3 grid;
    dim3 block;
};

std::ostream& operator << (std::ostream& o, dim3 const& d) {
    o << '(' << d.x << ',' << d.y << ')';
    return o;
}

void generateFrames(int const num_frames,
                    std::vector<MovingObject>& circles,
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

    for(int i_frame = 0; i_frame < num_frames; ++i_frame) {
        std::string const frame_string = "frame " + std::to_string(i_frame);
        std::cout << "Drawing " << frame_string << '\n';

        // Clear frame
        dim3 clear_block(16,16,1);
        dim3 clear_grid((FRAME_WIDTH+15)/16, (FRAME_HEIGHT+15)/16, 1);
        clearFrame<<<clear_grid, clear_block, 0, cuda_stream>>>(dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
        cuda_status = cudaGetLastError();
        cudaCheckSuccess(cuda_status, "Error launching 'clear' kernel in frame " + std::to_string(i_frame));

        // Draw circles
        int circle_counter = 0;
        for(auto &obj : circles) {
            cv::Rect2i const bb = obj.boundingBox();
            uchar3 const color{obj.shape().color()[0], obj.shape().color()[1], obj.shape().color()[2]};
            cv::Point2i const position = obj.position();
            int const radius = bb.size().width / 2;
            CudaThreadConfig const cuda_thread_config(bb.size());
            std::cout << "- Circle " << circle_counter
                      << " bb = " << cuda_thread_config.bb_size
                      << " grid = " << cuda_thread_config.grid
                      << " block = " << cuda_thread_config.block
                      << " fragmentation = " << cuda_thread_config.fragmentation()
                      << '\n';
            drawCircle<<<cuda_thread_config.grid, cuda_thread_config.block, 0, cuda_stream>>>(position.x, position.y, radius, color, dev_frame, FRAME_WIDTH, FRAME_HEIGHT);
            cuda_status = cudaGetLastError();
            cudaCheckSuccess(cuda_status, "Error launching kernel for circle " + std::to_string(circle_counter) + " in " + frame_string);
            obj.update();
            ++circle_counter;
        }

        // Fetch frame from GPU and write to img file
        cuda_status = cudaMemcpy((void*) frame.data, (void*) dev_frame, frame_num_bytes, cudaMemcpyDeviceToHost);
        cudaCheckSuccess(cuda_status, "Error copying data out of GPU in " + frame_string);
        std::ostringstream s;
        s << prefix << '-' << std::setw(3) << std::setfill('0') << i_frame << ".png";
        cv::imwrite(s.str(), frame);
    }

    cudaFree((void*) dev_frame);
}

int main(int const argc, char const *argv[])
{
    assert(argc == 3 && "Usage: video_frame_generator <frames> <prefix>-nnn.png");
    int const num_frames = std::atoi(argv[1]);
    std::vector<MovingObject> circles = generateRandomMovingCircles(25, FRAME_WIDTH, FRAME_HEIGHT);
    generateFrames(num_frames, circles, argv[2]);
    return 0;
}
