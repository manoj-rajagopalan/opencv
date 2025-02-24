/// Rotates input video to output

// Status: Doesn't quite work. Need to finish.
//
// The output is garbled in a tiled way.
// Discovered, close to giving up (out of time), that OpenCV's matrix dimensions are 
// transposed w.r.t. my expectations. That is, a 640 x 360 image is stored with
// matrix-dims 640 rows and 360 columns (whereas we image 640 being the number of
// columns so that the matrix visually aligns with the image). This is not a big problem,
// just that there were other things to get to by the time I discovered this.

#include <cassert>
#include <chrono>
#include <iostream>

#include "cuda.h"

#include "opencv2/videoio.hpp"

struct VideoProperties
{
    int num_frames;
    cv::Size frame_size;
    double fps;
    bool is_color;
    int cv_mat_type;
};

VideoProperties properties(cv::VideoCapture const& video_infile)
{
    VideoProperties video_properties;
    video_properties.num_frames = int(video_infile.get(cv::CAP_PROP_FRAME_COUNT));
    video_properties.frame_size.height = int(video_infile.get(cv::CAP_PROP_FRAME_WIDTH));
    video_properties.frame_size.width = int(video_infile.get(cv::CAP_PROP_FRAME_HEIGHT));
    video_properties.fps = video_infile.get(cv::CAP_PROP_FPS);
    video_properties.cv_mat_type = int(video_infile.get(cv::CAP_PROP_FORMAT));
    video_properties.is_color = true;
    return video_properties;
}

class Image {
public:
    Image() = delete;

    __device__
    Image(uchar3 *data, int width, int height)
    : data_(data), width_(width), height_(height)
    {}

    __device__
    uchar3& operator()(int x, int y) {
        return data_[y * width_ + x];
    }

    __device__
    uchar3 const& operator()(int x, int y) const {
        return data_[y * width_ + x];
    }

private:
    uchar3 *data_;
    int width_;
    int height_;
};

constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 32;

__global__
void cudaTransposeImage(uchar3 *const out_image,
                        uchar3 *const in_image,
                        int const width,
                        int const height)
{
    Image const in(in_image, width, height);
    Image out(out_image, height, width);

    __shared__ uchar3 sm_sub_img_data[THREADS_Y][THREADS_X+5]; // improve bank collisions
    Image sm_sub_img(&sm_sub_img_data[0][0], THREADS_X+5, THREADS_Y);

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x < width && y < height) {
        // sm_sub_img(threadIdx.x, threadIdx.y) = in(x,y);
        // __syncthreads();
        // out(height-1-y,x) = sm_sub_img(threadIdx.x, threadIdx.y);
        out(height-1-y,x) = in(x,y);
    }
    return;
    // x = (blockDim.y-1-blockIdx.y) * blockDim.y + threadIdx.x;
    // y = blockIdx.x * blockDim.x + threadIdx.y;
    // int height_residue = height % blockDim.y;
    // if(height_residue == 0) { height_residue = blockDim.y; }
    // if(x < height && y < width) {
    //     out(x,y) = sm_sub_img(threadIdx.y, height_residue-1 - threadIdx.x);
    // }
}

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << ": " << message << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

int main(int const argc, char const *const argv[])
{
    assert(argc == 3 && "Usage: video_rotate <infile> <outfile>");

    cv::VideoCapture video_infile;
    bool status = video_infile.open(argv[1]);
    assert(status && "Unable to open input file");
    VideoProperties const video_properties = properties(video_infile);
    std::cout << "Image dims: " << video_properties.frame_size.width << " x " << video_properties.frame_size.height;
    std::cout << " [" << video_properties.num_frames << "]" << std::endl;
    std::cout << "Input mat type = " << video_properties.cv_mat_type << std::endl;
    // assert(video_properties.cv_mat_type == CV_8UC3);

    cv::VideoWriter video_outfile;
    status = video_outfile.open(argv[2], cv::VideoWriter::fourcc('H','2','6','4'),
                                video_properties.fps, video_properties.frame_size, video_properties.is_color);

    // CPU image frames
    cv::Mat in_frame;
    // status = video_infile.retrieve(in_frame, 0);
    video_infile >> in_frame;
    std::cout << "in_frame.size: " << in_frame.size[0] << " x " << in_frame.size[1]
              << " ( " << in_frame.rows << 'x' << in_frame.cols << " )" << std::endl;
    // assert(status && "Unable to retrieve frame #0");
    cv::Mat out_frame(in_frame.cols, in_frame.rows, in_frame.type());
    std::cout << "Frame 0 cv_mat_type = " << in_frame.type() << std::endl;

    // GPU image frames
    uchar3 *gpu_in_frame;
    uchar3 *gpu_out_frame;
    int const num_pixels = video_properties.frame_size.width * video_properties.frame_size.height;
    int const num_bytes_per_frame = num_pixels * 3;
    std::cout << "num_pixels = " << num_pixels << std::endl;
    cudaError_t cuda_status = cudaMalloc((void**) &gpu_in_frame, num_bytes_per_frame);
    assert(cuda_status == cudaSuccess && "Unable to allocate input frame on GPU");
    cuda_status = cudaMalloc((void**) &gpu_out_frame, num_bytes_per_frame);
    assert(cuda_status == cudaSuccess && "Unable to allocate output frame on GPU");

    dim3 const blockDim(THREADS_X, THREADS_Y);
    int const width = video_properties.frame_size.width;
    int const height = video_properties.frame_size.height;
    dim3 const gridDim((width+THREADS_X-1)/THREADS_X, (height+THREADS_Y-1)/THREADS_Y);
    std::cout << "gridDim  = " << gridDim.x << ',' << gridDim.y << std::endl;
    std::cout << "blockDim = " << blockDim.x << ',' << blockDim.y << std::endl;

    // Process frames
    for(int i_frame = 1; i_frame < video_properties.num_frames; ++i_frame) {
	std::chrono::time_point<std::chrono::steady_clock> t_begin = std::chrono::steady_clock::now();

        status = video_infile.read(in_frame);
        assert(status && "Unable to read input frame from video");

        assert(in_frame.isContinuous());
        
        cuda_status = cudaMemcpy((void*) gpu_in_frame, (void*) in_frame.data, num_bytes_per_frame, cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess && "Unable to copy frame into GPU");

	cuda_status = cudaGetLastError();
	cudaCheckSuccess(cuda_status, "Error before kernel launch");
    cudaTransposeImage<<<gridDim, blockDim>>>(gpu_out_frame, gpu_in_frame, width, height);
        
	cuda_status = cudaGetLastError();
	cudaCheckSuccess(cuda_status, "Unable to launch kernel");
	cudaDeviceSynchronize();
        cuda_status = cudaMemcpy((void*) out_frame.data, (void*) gpu_out_frame, num_bytes_per_frame, cudaMemcpyDeviceToHost);
        cudaCheckSuccess(cuda_status, "Unable to copy frame out of GPU");
        
	std::chrono::duration<float> const duration = std::chrono::nanoseconds(std::chrono::steady_clock::now() - t_begin);
        std::cout << "Frame " << i_frame  << "  " << duration.count() << " ns" << std::endl;
        video_outfile.write(out_frame);
    }

    return 0;
}
