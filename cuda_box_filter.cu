#include <cuda.h>

#include <opencv2/imgcodecs.hpp>

#include <cassert>
#include <iostream>
#include <string>

__global__
void boxFilterCuda(uchar3 *output_rgb_img,
                   uchar3 *input_rgb_img,
                   int input_img_width,
                   int input_img_height,
                   int k_width, // must be odd
                   int k_height /* must be odd*/)
{
    int x = threadIdx.x % input_img_height;
    int y = threadIdx.x / input_img_height;

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

            const uchar3 sample = *(input_rgb_img + 3 * (yj * input_img_height + xi));
            sum = sum + make_uint3(sample.x, sample.y, sample.z);
        }
    }
    const float3 sumf = make_float3(sum.x, sum.y, sum.z) / make_float3(k_size, k_size, k_size);
    output_rgb_img[threadIdx.x] = make_uchar3(__float2int_rd(sumf.x), __float2int_rd(sumf.y), __float2int_rd(sumf.z));
}

int main(int argc, char *argv[])
{
    // Read in and check input image
    assert(argc == 2 && "Provide image file to filter");
    const std::string input_img_filename(argv[1]);
    cv::Mat input_img = cv::imread(input_img_filename);
    size_t const input_img_num_pixels = input_img.rows * input_img.cols;
    size_t const input_img_num_bytes = input_img_num_pixels * input_img.elemSize();
    assert(input_img.isContinuous());
    assert(input_img.elemSize() == (input_img.channels() * input_img.elemSize1()));
    assert(input_img.elemSize1() == 1 && "Can only work on RGB images");
    assert(input_img.channels() == 3 && "Can only work on RGB images");

    // Allocate output image from CUDA and check
    cv::Mat output_img_cuda(input_img.size(), input_img.depth());
    assert(input_img.isContinuous());
    assert(input_img.elemSize() == (input_img.channels() * input_img.elemSize1()));
    assert(input_img.elemSize1() == 1 && "Can only work on RGB images");
    assert(input_img.channels() == 3 && "Can only work on RGB images");

    cudaError_t status = cudaSuccess;
    
    // Allocate input image on GPU and copy over
    uchar3 *dev_input_img;
    status = cudaMalloc((void**) &dev_input_img, input_img_num_bytes);
    assert(status == cudaSuccess && "input_img cudaMalloc failed");
    status = cudaMemcpy((void*) dev_input_img, (void*) input_img.data, input_img_num_bytes, cudaMemcpyHostToDevice);
    assert(status == cudaSuccess && "input_img cudaMemcpy failed");

    // Allocate output image on GPU
    uchar3 *dev_output_img;
    status = cudaMalloc((void**) &dev_output_img, input_img_num_bytes);
    assert(status == cudaSuccess && "input_img cudaMalloc failed");

    // Run kernel
    boxFilterCuda<<<1, input_img_num_pixels>>>(dev_output_img, dev_input_img, input_img.rows, input_img.cols, 3, 3);

    // Copy out of GPU
    status = cudaMemcpy((void*) output_img_cuda.data, dev_output_img, input_img_num_bytes, cudaMemcpyDeviceToHost);
    assert(status == cudaSuccess && "output_img cudaMemcpy failed");

    // Write to file
    const auto dot_pos = input_img_filename.rfind('.');
    const std::string output_filename =
        input_img_filename.substr(0, dot_pos) + "-box_filter_cuda" + input_img_filename.substr(dot_pos);
    cv::imwrite(output_filename, output_img_cuda);


    // Release resources
    cudaFree(dev_input_img);
    cudaFree(dev_output_img);

    return 0;
}
