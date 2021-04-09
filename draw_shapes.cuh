#ifndef MANOJ_OPENCV_DRAW_SHAPES_CUH
#define MANOJ_OPENCV_DRAW_SHAPES_CUH

#include "cuda.h"

__global__ void clearFrame(uchar3 *const frame, int const frame_width, int const frame_height);

__global__ void drawCircle(int const x0, int const y0, int const radius, uchar3 const color,
                           uchar3 *const frame, int const frame_width, int const frame_height);

__global__ void drawRect(int const x0, int const y0,
                         int const x1, int const y1,
                         uchar3 const color,
                         uchar3 *const frame, int const frame_width, int const frame_height);

#endif // MANOJ_OPENCV_DRAW_SHAPES_CUH
