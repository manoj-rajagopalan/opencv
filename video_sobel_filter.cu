/// Processes input video file and writes to output

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
    video_properties.frame_size.width = int(video_infile.get(cv::CAP_PROP_FRAME_WIDTH));
    video_properties.frame_size.height = int(video_infile.get(cv::CAP_PROP_FRAME_HEIGHT));
    video_properties.fps = video_infile.get(cv::CAP_PROP_FPS);
    video_properties.cv_mat_type = int(video_infile.get(cv::CAP_PROP_FORMAT));
    video_properties.is_color = true;
    return video_properties;
}


__device__ float3 getPixel(uchar3 const *const image,
		           int const width,
		           int const height,
		           int const x,
			   int const y)
{
	if(x < 0 || x >= width || y < 0 || y >= height) {
		return make_float3(0,0,0);
	} else {
		int const pixel_index = y * width + x;
		uchar3 const pixel = image[pixel_index];
		return make_float3(pixel.x, pixel.y, pixel.z);
	}
}

__device__ float norm_sqr(float3 const v)
{
	return v.x*v.x + v.y*v.y + v.z*v.z;
}

__device__ float3& operator += (float3& a, float3 const b)
{
	a.x += b.x;
	a.y += b.y;
	a.z += b.z;
	return a;
}

__device__ float3& operator -= (float3& a, float3 const b)
{
	a.x -= b.x;
	a.y -= b.y;
	a.z -= b.z;
	return a;
}

__device__ float3 operator *(float3 const& a, int n)
{
	return make_float3(n*a.x, n*a.y, n*a.z);
}

__global__
void cudaSobelFilter(uchar3 *const out_image,
                       const uchar3 *const in_image,
                       int const num_pixels,
                       int const image_width,
                       int const image_height)
{
    int const tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < num_pixels) {
        int const x = tid % image_width;
        int const y = tid / image_width;

	float3 gx = make_float3(0,0,0), gy = make_float3(0,0,0);
	float3 const ne = getPixel(in_image, image_width, image_height, x-1, y-1);
	gx -= ne;
	gy -= ne;
	float3 const nw = getPixel(in_image, image_width, image_height, x+1, y-1);
	gx += nw;
	gy -= nw;
	float3 const n = getPixel(in_image, image_width, image_height, x, y-1);
	gy -= n*2;
	float3 const s = getPixel(in_image, image_width, image_height, x, y+1);
	gy -= s*2;
	float3 const e = getPixel(in_image, image_width, image_height, x-1, y);
	gx -= e*2;
	float3 const w = getPixel(in_image, image_width, image_height, x+1, y);
	gx += w*2;
	float3 const se = getPixel(in_image, image_width, image_height, x-1, y+1);
	gx -= se;
	gy += se;
	float3 const sw = getPixel(in_image, image_width, image_height, x+1, y+1);
	gx += sw;
	gy += sw;

	float v = sqrt(norm_sqr(gx) + norm_sqr(gy));
        uchar3 pixel = make_uchar3(int(v), int(v), int(v));
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

int main(int const argc, char const *const argv[])
{
    assert(argc == 3 && "Usage: video_box_blur_filterer <infile> <outfile>");

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
    // assert(status && "Unable to retrieve frame #0");
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
    for(int i_frame = 1; i_frame < video_properties.num_frames; ++i_frame) {
	std::chrono::time_point<std::chrono::steady_clock> t_begin = std::chrono::steady_clock::now();

        status = video_infile.read(in_frame);
        assert(status && "Unable to read input frame from video");

        assert(in_frame.isContinuous());
        
        cuda_status = cudaMemcpy((void*) gpu_in_frame, (void*) in_frame.data, num_bytes_per_frame, cudaMemcpyHostToDevice);
        assert(cuda_status == cudaSuccess && "Unable to copy frame into GPU");

	cuda_status = cudaGetLastError();
	cudaCheckSuccess(cuda_status, "Error before kernel launch");
        cudaSobelFilter<<<(num_pixels + 1023)/1024, 1024>>>(gpu_out_frame, gpu_in_frame, num_pixels,
                                                              video_properties.frame_size.width, video_properties.frame_size.height);
        
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
