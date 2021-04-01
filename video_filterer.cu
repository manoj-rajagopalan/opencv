/// Processes input video file and writes to output

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

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
    video_properties.frame_size.width = int(video_infile.get(cv::CAP_PROP_FRAME_WIDTH));
    video_properties.frame_size.height = int(video_infile.get(cv::CAP_PROP_FRAME_HEIGHT));
    video_properties.fps = video_infile.get(cv::CAP_PROP_FPS);
    video_properties.cv_mat_type = int(video_infile.get(cv::CAP_PROP_FORMAT));
    video_properties.is_color = true;
    return video_properties;
}

__constant__ float filter_gpu[25];

__global__
void cudaFilter(uchar3 *const out_image,
                const uchar3 *const in_image,
                int const num_pixels,
                int const image_width,
                int const image_height,
                int const filter_width,
                int const filter_height)
{
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_pixels) {
        int const x = tid % image_width;
        int const y = tid / image_width;
        
        float sum_x = 0.0f, sum_y = 0.0f, sum_z = 0.0f;
        int const window_right = filter_width / 2;
        int const window_left = -window_right;
        int const window_bottom = filter_height / 2;
        int const window_top = -window_bottom;
        for(int i = window_top; i <= window_bottom; ++i) {
            int yi = y + i;
            yi = yi < 0 ? 0 : (yi >= image_height ? (image_height-1) : yi);

            for(int j = window_left; j <= window_right; ++j) {
                int xj = x + j;
                xj = xj < 0 ? 0 : (xj >= image_width ? (image_width-1) : xj);

                int const pixel_index = yi * image_width + xj;
                uchar3 const pixel = in_image[pixel_index];

                float const filter_coeff = filter_gpu[(i - window_top) * filter_width + (j - window_left)];
                sum_x += filter_coeff * pixel.x;
                sum_y += filter_coeff * pixel.y;
                sum_z += filter_coeff * pixel.z;
            }
        }
        uchar3 pixel = make_uchar3(int(sum_x), int(sum_y), int(sum_z));
        out_image[tid] = pixel;
    }

}

void cudaCheckSuccess(cudaError_t const cuda_status, std::string const& message)
{
    if(cudaSuccess != cuda_status) {
        std::cout << "CUDA ERROR " << cuda_status << ": " << message << std::endl;
        std::cout << "- " << cudaGetErrorName(cuda_status) << ": " << cudaGetErrorString(cuda_status) << std::endl;
    }
}

class Filter
{
public:
    Filter() {}
    ~Filter() {}

    int width() const { return width_; }
    int height() const { return height_; }
    int area() const { return int(data_.size()); }

    float const* data() const { return data_.data(); }

    void loadFromFile(std::string const& file_name) {
        std::ifstream file(file_name.c_str());
        assert(file && "Unable to open filter file");
        while(file.good()) {
            ++height_;
            std::string line;
            std::getline(file, line);
            std::istringstream line_input(line);
            int line_width = 0;
            float param = 0.0f;
            while(line_input) {
                param = std::nanf("");
                line_input >> param;
                if(std::isnan(param)) {
                    break;
                }
                data_.push_back(param);
                ++line_width;
            }
            assert(width_ == 0 || line_width == width_);
            width_ = line_width;
        }
        assert((width_ * height_) == int(data_.size()) && "Error parsing filter");
    }

    void print(std::ostream& o, std::string const& line_begin = "") const {
        auto data_iter = data_.cbegin();
        for(int i = 0; i < height_; ++i) {
            o << line_begin;
            for(int j = 0; j < width_; ++j) {
                o << *data_iter << ' ';
                ++data_iter;
            }
            o << '\n';
        }
    }

private:
    int width_ = 0;
    int height_ = 0;
    std::vector<float> data_;
};

int main(int const argc, char const *const argv[])
{
    assert(argc == 4 && "Usage: video_filterer <infile> <outfile> <filter file (row major order, max 25 elements)");

    cv::VideoCapture video_infile;
    bool status = video_infile.open(argv[1]);
    assert(status && "Unable to open input file");
    VideoProperties const video_properties = properties(video_infile);
    std::cout << "Input mat type = " << video_properties.cv_mat_type << std::endl;
    // assert(video_properties.cv_mat_type == CV_8UC3);

    cv::VideoWriter video_outfile;
    status = video_outfile.open(argv[2], cv::VideoWriter::fourcc('H','2','6','4'),
                                video_properties.fps, video_properties.frame_size, video_properties.is_color);

    Filter filter_cpu;
    filter_cpu.loadFromFile(argv[3]);
    std::cout << "Filter dims: " << filter_cpu.width() << 'x' << filter_cpu.height() << " = " << filter_cpu.area() << std::endl;
    filter_cpu.print(std::cout, "\t");
    cudaMemcpyToSymbol(filter_gpu, (const void*) filter_cpu.data(), filter_cpu.area(), /*offset=*/ 0, cudaMemcpyHostToDevice);

    // CPU image frames
    cv::Mat in_frame;
    status = video_infile.retrieve(in_frame, 0);
    cv::Mat out_frame(in_frame.size[0], in_frame.size[1], in_frame.type());
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

    // Process frames
    for(int i_frame = 0; i_frame < video_properties.num_frames; ++i_frame) {
        std::cout << "Frame " << i_frame << std::endl;

        status = video_infile.read(in_frame);
        assert(status && "Unable to read input frame from video");

        assert(in_frame.isContinuous());
        
        cuda_status = cudaMemcpy((void*) gpu_in_frame, (void*) in_frame.data, num_bytes_per_frame, cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess && "Unable to copy frame into GPU");

        cudaFilter<<<(num_pixels + 1023)/1024, 1024>>>(gpu_out_frame, gpu_in_frame, num_pixels,
                                                        video_properties.frame_size.width, video_properties.frame_size.height,
                                                        filter_cpu.width(), filter_cpu.height());
        cuda_status = cudaGetLastError();
        cudaCheckSuccess(cuda_status, "Error launching kernel");
        
        cuda_status = cudaMemcpy((void*) out_frame.data, (void*) gpu_out_frame, num_bytes_per_frame, cudaMemcpyDeviceToHost);
        cudaCheckSuccess(cuda_status, "Unable to copy frame out of GPU");
        
        video_outfile.write(out_frame);
    }

    return 0;
}