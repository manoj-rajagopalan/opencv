#include "cpu_gpu_image.hpp"
#include "cuda_histogram.cuh"
#include "draw_shapes.cuh"

#include <cmath>
#include <sstream>
#include "opencv2/imgproc.hpp"

void fillHistograms(int i_frame,
                    float r_histogram[256],
                    float g_histogram[256],
                    float b_histogram[256])
{
    for(int i = 0; i < 256; ++i) {
        double const theta_deg = (i / 256.0) * 180 + i_frame;
        double const theta_rad = theta_deg * M_PI  / 180;
        r_histogram[i] = std::max(0.0, 0.5 + 0.5 * std::cos(theta_rad));
        g_histogram[i] = std::max(0.0, 0.5 + 0.5 * std::sin(theta_rad));
        b_histogram[i] = 0.4 + 0.3 * std::sin(2 * theta_rad);
    }
}

int main(int argc, char *argv[])
{
    assert(argc == 2 && "Usage: histogram_test <output pattern>-%03d.png")
    CpuGpuImage cpu_gpu_image(600, 480);
    cpu_gpu_image.cpuMat() = 0; // clear to black

    // clear
    dim3 const cuda_block(32,32,1);
    dim3 const cuda_grid((cpu_gpu_image.width()+31)/32, (cpu_gpu_image.height()+31)/32, 1);
    clearFrame<<<cuda_grid, cuda_block>>>(cpu_gpu_image.gpuPtr(), cpu_gpu_image.width(), cpu_gpu_image.height());

    float r_histogram[256];
    float g_histogram[256];
    float b_histogram[256];

    Frame const gpu_frame(cpu_gpu_image.gpuPtr(), cpu_gpu_image.width(), cpu_gpu_image.height());
    for(int i_frame = 0; i_frame < 100; ++i_frame) {
        fillHistograms(i_frame, r_histogram, g_histogram, b_histogram);
        drawHistogram<<<1,258>>>(gpu_frame, 0, 0, 1, 100, r_histogram, g_histogram, b_histogram);
        cpu_gpu_image.copyFromGpu();
        std::ostringstream outfile_stream;
        outfile_stream << argv[1] << '-' << std::setw(3) << std::setfill('0') << ".png";
        cv::imwrite(outfile_stream.str(), cpu_gpu_image.cpuMat());
    }

    return 0;
}