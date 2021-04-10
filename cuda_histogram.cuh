#ifndef MANOJ_CUDA_OPENCV_HISTOGRAM_CUH
#define MANOJ_CUDA_OPENCV_HISTOGRAM_CUH

#include "cuda.h"

struct Frame
{
    uchar3 *const ptr;
    int32_t width;
    int32_t height;
};

__global__ void resetHistograms(int32_t r_histogram[256],
                                int32_t g_histogram[256],
                                int32_t b_histogram[256]);

__global__ void rgbHistogramSinglePixGlobal(Frame const frame,
                                            int32_t r_histogram[256],
                                            int32_t g_histogram[256],
                                            int32_t b_histogram[256]);

__global__ void rgbHistogramSinglePixShMem(Frame const frame,
                                           int32_t r_histogram[256],
                                           int32_t g_histogram[256],
                                           int32_t b_histogram[256]);

__global__ void drawHistogram(Frame const frame,
                              int32_t const x,
                              int32_t const y,
                              int32_t const hist_width_per_bin,
                              int32_t const hist_height,
                              int32_t r_histogram[256],
                              int32_t g_histogram[256],
                              int32_t b_histogram[256]);

#endif //  MANOJ_CUDA_OPENCV_HISTOGRAM_CUH
