#include <cuda.h>
#include <math.h>

#include <opencv2/imgcodecs.hpp>

#include <cassert>
#include <iostream>
#include <string>

using std::cout;
using std::endl;

__device__ __forceinline__
uint3 operator + (const uint3 a, const uint3 b)
{
    uint3 c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    c.z = a.z + b.z;
    return c;
}

__device__ __forceinline__
float3 operator / (const float3 a, const float3 b)
{
    float3 c;
    c.x = a.x / b.x;
    c.y = a.y / b.y;
    c.z = a.z / b.z;
    return c;
}

__global__
void writePattern(uchar3 *const output_rgb_img)
{
    // output_rgb_img[threadIdx.x] = make_uchar3(threadIdx.x % 256, (128 + threadIdx.x) % 256, 255 - (threadIdx.x % 256));
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    output_rgb_img[n] = make_uchar3(threadIdx.x, (128 + threadIdx.x) % 256, 255 - threadIdx.x);
}

__global__
void copyCuda(uchar3 *output_rgb_img,
              uchar3 *input_rgb_img,
              int const num_pixels)
{
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < num_pixels) {
        output_rgb_img[n] = input_rgb_img[n];
    }
}

__global__
void boxFilterCuda(uchar3 *const __restrict__ output_rgb_img,
                   uchar3 const *const __restrict__ input_rgb_img,
                   int input_img_width,
                   int input_img_height,
                   int k_width, // must be odd
                   int k_height /* must be odd*/)
{
    int const num_pixels = input_img_width * input_img_height;
    int const n = blockIdx.x * blockDim.x + threadIdx.x;
    if(n < num_pixels) {
        int x = n % input_img_width;
        int y = n / input_img_width;

        const int k_size = k_width * k_height;
        k_width /= 2;
        k_height /= 2;

        uint3 sum{0,0,0};
        for(int j = -k_height; j <= k_height; ++j) {
            int yj = y + j;
            yj = yj < 0 ? 0 : (yj >= input_img_height ? (input_img_height-1) : yj); // clamp

            for(int i = -k_width; i <= k_width; ++i) {
                int xi = x + i;
                xi = xi < 0 ? 0 : (xi >= input_img_width ? (input_img_width-1) : xi); // clamp

                const uchar3 sample = *(input_rgb_img + (yj * input_img_width + xi));
                sum = sum + make_uint3(sample.x, sample.y, sample.z);
            }
        }
        const float3 sumf = make_float3(sum.x, sum.y, sum.z) / make_float3(k_size, k_size, k_size);
        output_rgb_img[n] = make_uchar3(__float2int_rd(sumf.x), __float2int_rd(sumf.y), __float2int_rd(sumf.z));
    }
}

int main(int argc, char *argv[])
{
    // Read in and check input image
    assert(argc == 2 && "Provide image file to filter");
    const std::string input_img_filename(argv[1]);
    cv::Mat input_img = cv::imread(input_img_filename);
    cout << "Input image dims: " << input_img.rows << " x " << input_img.cols << endl;
    cout << "Input image type: " << input_img.type() << endl;
    size_t const input_img_num_pixels = input_img.rows * input_img.cols;
    cout << "Input image num pixels = " << input_img_num_pixels << endl;
    cout << "Input image elemtSize = " << input_img.elemSize() << endl;
    size_t const input_img_num_bytes = input_img_num_pixels * input_img.elemSize();
    cout << "Input image size = " << input_img_num_bytes << " B" << endl;
    assert(input_img.isContinuous());
    assert(input_img.elemSize() == (input_img.channels() * input_img.elemSize1()));
    assert(input_img.elemSize1() == 1 && "Can only work on RGB images");
    assert(input_img.channels() == 3 && "Can only work on RGB images");

    // Allocate output image from CUDA and check
    cv::Mat output_img(input_img.size(), input_img.type());
    cout << "Output image dims: " << input_img.rows << " x " << input_img.cols << endl;
    cout << "Output image type: " << output_img.type() << endl;
    assert(output_img.isContinuous());
    assert(output_img.elemSize() == (output_img.channels() * output_img.elemSize1()));
    assert(output_img.elemSize1() == 1 && "Can only work on RGB images");
    assert(output_img.channels() == 3 && "Can only work on RGB images");

    cudaError_t status = cudaSuccess;
    
    // Allocate input image on GPU and copy over
    uchar3 *dev_input_img;
    status = cudaMalloc((void**) &dev_input_img, input_img_num_bytes);
    assert(status == cudaSuccess && "input_img cudaMalloc failed");
    status = cudaMemcpy((void*) dev_input_img, (void*) input_img.data, input_img_num_bytes, cudaMemcpyHostToDevice);
    assert(status == cudaSuccess && "input_img cudaMemcpy failed");

    // Allocate output image on GPU
    uchar3 *dev_output_img = 0x0;
    status = cudaMalloc((void**) &dev_output_img, input_img_num_bytes);
    assert(status == cudaSuccess && "input_img cudaMalloc failed");
    cout << "dev_output_img allocated at " << std::hex << dev_output_img << endl;

    // Run kernel
    boxFilterCuda<<<(input_img_num_pixels + 999)/1000, 1000>>>(dev_output_img, dev_input_img, input_img.rows, input_img.cols, 5, 5);
    // copyCuda<<<(input_img_num_pixels + 999)/1000, 1000>>>(dev_output_img, dev_input_img, input_img_num_pixels);
    // writePattern<<<1, 256>>>(dev_output_img, input_img_num_pixels);

    // Copy out of GPU
    status = cudaMemcpy((void*) output_img.data, dev_output_img, input_img_num_bytes, cudaMemcpyDeviceToHost);
    assert(status == cudaSuccess && "output_img cudaMemcpy failed");

    status = cudaDeviceSynchronize();
    assert(status == cudaSuccess && "cudaDeviceSynchronize failed");

    // Write to file
    const auto dot_pos = input_img_filename.rfind('.');
    const std::string output_filename =
        input_img_filename.substr(0, dot_pos) + "-box_filter_cuda" + input_img_filename.substr(dot_pos);
    cv::imwrite(output_filename, output_img);


    // Release resources
    cudaFree(dev_input_img);
    cudaFree(dev_output_img);

    return 0;
}
